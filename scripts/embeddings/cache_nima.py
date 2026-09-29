#!/usr/bin/env python3
"""Generate a resumable, immutable pretrained NIMA scalar cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.embeddings.feature_contract import write_contract
from image_aesthetics.embeddings.nima_provider import NimaExtractor, nima_contract, prepare_nima_image
from image_aesthetics.embeddings.preprocess import load_manifest
from image_aesthetics.embeddings.ten_signal_runner import RunnerConfig, TenSignalRunner, source_fingerprints_sha256


def dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify(cache: Path, manifest: Path, image_root: Path, contract: Path) -> dict:
    command = [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"), "verify",
               "--cache-dir", str(cache), "--manifest", str(manifest),
               "--image-root", str(image_root), "--contract", str(contract)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    value = json.loads(result.stdout)
    if result.returncode or value.get("errors"):
        raise RuntimeError(f"strict verification failed: {value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-workers", type=int, default=6)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--max-new-images", type=int, default=0)
    args = parser.parse_args()

    manifest = args.manifest.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    items = load_manifest(manifest, image_root=image_root)
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fingerprints = source_fingerprints_sha256(TenSignalRunner._fingerprint_sources(items))
    contract_path = output / "feature-contract.json"
    identity = write_contract(contract_path, input_manifest_sha256=manifest_sha,
                              source_fingerprints_sha256=fingerprints, contract=nima_contract())
    cache = output / "cache"
    runner = TenSignalRunner(
        RunnerConfig(cache_dir=cache, batch_size=args.batch_size, decode_workers=args.decode_workers,
                     image_size=299, shard_size=args.shard_size, max_new_images=args.max_new_images),
        batch_extractor=NimaExtractor(), native_prepare_fn=prepare_nima_image,
    )
    started = time.perf_counter()
    result = runner.run(items, contract_path, run_id="nima-initial")
    dump(output / "initial-result.json", result)
    if args.max_new_images and result.get("state") != "completed":
        dump(output / "report.json", {"state": "partial", "cache_identity": identity,
                                      "image_count": len(items), "result": result})
        return 0
    checked = verify(cache, manifest, image_root, contract_path)
    dump(output / "strict-verify.json", checked)
    resumed = runner.run(items, contract_path, run_id="nima-resume")
    dump(output / "resume-result.json", resumed)
    if resumed.get("decoded_images") or resumed.get("inference_batches"):
        raise RuntimeError(f"resume performed work: {resumed}")
    report = {"state": "completed", "cache_identity": identity, "image_count": len(items),
              "elapsed_seconds": time.perf_counter() - started, "result": result,
              "resume": resumed, "strict_verify": checked}
    dump(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
