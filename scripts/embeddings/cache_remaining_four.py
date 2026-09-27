#!/usr/bin/env python3
"""Generate the verified CLIP, LAION-CLIP, and dual-MUSIQ feature shard set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.embeddings.feature_contract import write_contract
from image_aesthetics.embeddings.preprocess import load_manifest
from image_aesthetics.embeddings.remaining_providers import (
    NATIVE_MAX_SIDE, REMAINING_FOUR, RemainingFourExtractor,
    prepare_remaining_images, remaining_contract,
)
from image_aesthetics.embeddings.ten_signal_runner import (
    RunnerConfig, TenSignalRunner, source_fingerprints_sha256,
)


def _dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verify(cache: Path, manifest: Path, image_root: Path, contract_path: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"), "verify",
               "--cache-dir", str(cache), "--manifest", str(manifest),
               "--image-root", str(image_root), "--contract", str(contract_path)]
    result = subprocess.run(command, cwd=ROOT, env=env, check=False, capture_output=True, text=True)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"strict verify failed: {result.stderr or result.stdout}") from None
    if result.returncode or payload.get("errors"):
        raise RuntimeError(f"strict verify failed: {payload}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="optional deterministic random sample size for an isolated smoke run")
    parser.add_argument("--seed", type=int, default=20260927,
                        help="seed for deterministic representative smoke selection")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-new-images", type=int, default=0,
                        help="process at most this many pending rows, then exit resumably; zero means all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_images < 0:
        raise SystemExit("--max-new-images must be zero or positive")
    source_manifest = args.manifest.resolve(strict=True)
    image_root = args.image_root.resolve(strict=True)
    rows = [json.loads(line) for line in source_manifest.read_text().splitlines() if line.strip()]
    if args.limit < 0 or (args.limit and args.limit > len(rows)):
        raise SystemExit("--limit must be zero or no larger than the source manifest")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.limit:
        indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.limit))
        chosen = [rows[index] for index in indices]
        manifest = output / "smoke-manifest.jsonl"
        encoded = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in chosen)
        if manifest.exists() and manifest.read_text() != encoded:
            raise SystemExit(f"refusing to replace a different smoke manifest: {manifest}")
        if not manifest.exists():
            manifest.write_text(encoded)
    else:
        manifest = source_manifest

    items = load_manifest(manifest, image_root=image_root)
    config = remaining_contract()
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fingerprints_sha = source_fingerprints_sha256(TenSignalRunner._fingerprint_sources(items))
    contract_path = output / "feature-contract.json"
    identity = write_contract(contract_path, input_manifest_sha256=manifest_sha,
                              source_fingerprints_sha256=fingerprints_sha, contract=config)
    cache = output / "cache"
    report_path = output / "report.json"
    report = {
        "schema_version": 1,
        "state": "starting",
        "cache_group": "remaining-four-v1",
        "dataset": output.parent.parent.name,
        "source_manifest": str(source_manifest),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "image_root": str(image_root),
        "source_fingerprints_sha256": fingerprints_sha,
        "cache_identity": identity,
        "image_count": len(items),
        "selection": {"kind": "deterministic_random_sample" if args.limit else "all_rows",
                      "limit": args.limit or None, "seed": args.seed if args.limit else None},
        "active_signals": list(REMAINING_FOUR),
        "blocked_signals": ["nima"],
        "musiq_preprocessing": {
            "input_long_side_cap": NATIVE_MAX_SIDE,
            "aspect_ratio_preserved": True,
            "internal_long_side_scales": [224, 384],
            "original_stream_max_patches": 256,
        },
    }
    if report_path.exists():
        old = json.loads(report_path.read_text())
        for key in ("cache_group", "manifest_sha256", "source_fingerprints_sha256", "cache_identity", "image_count"):
            if old.get(key) != report[key]:
                raise SystemExit(f"refusing to reuse output with changed {key}: {output}")
    else:
        _dump(report_path, report)

    try:
        extractor = RemainingFourExtractor()
        runner = TenSignalRunner(
            RunnerConfig(cache_dir=cache, batch_size=args.batch_size,
                         decode_workers=args.decode_workers, image_size=224,
                         shard_size=args.shard_size, max_new_images=args.max_new_images),
            batch_extractor=extractor,
            native_prepare_fn=prepare_remaining_images,
        )
        initial_started = time.perf_counter()
        initial = runner.run(items, contract_path, run_id="remaining-four-initial")
        initial_elapsed = time.perf_counter() - initial_started
        _dump(output / "initial-result.json", initial)
        if args.max_new_images and initial.get("state") != "completed":
            if initial.get("failed_images", 0):
                raise RuntimeError(f"chunk ended with failed images: {initial}")
            report.update(state="partial", last_chunk=initial,
                          chunk_limit=args.max_new_images,
                          cache_completed_images=initial.get("completed_images"))
            _dump(report_path, report)
            print(json.dumps({"state": "partial", "images": len(items),
                              "cache_completed_images": initial.get("completed_images"),
                              "chunk_limit": args.max_new_images,
                              "output": str(output), "initial": initial}, indent=2))
            return 0
        verify = _verify(cache, manifest, image_root, contract_path)
        _dump(output / "strict-verify.json", verify)
        resume_started = time.perf_counter()
        resumed = runner.run(items, contract_path, run_id="remaining-four-resume")
        resume_elapsed = time.perf_counter() - resume_started
        _dump(output / "resume-result.json", resumed)
        resume_verify = _verify(cache, manifest, image_root, contract_path)
        _dump(output / "resume-verify.json", resume_verify)
        status_proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"),
             "status", "--cache-dir", str(cache)],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True, check=True,
        )
        failures_proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"),
             "failures", "--cache-dir", str(cache)],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True, check=True,
        )
        status = json.loads(status_proc.stdout)
        failures = json.loads(failures_proc.stdout)
        _dump(output / "status.json", status)
        _dump(output / "failures.json", failures)
        metrics = {
            "images": len(items),
            "initial_elapsed_seconds": initial_elapsed,
            "initial_images_per_second": len(items) / initial_elapsed if initial_elapsed else 0.0,
            "resume_elapsed_seconds": resume_elapsed,
            "provider_seconds": initial.get("provider_seconds", {}),
            "provider_failures": initial.get("provider_failures", {}),
            "resident_memory_gb": initial.get("resident_memory_gb"),
            "swap_delta_gb": initial.get("swap_delta_gb"),
            "decode_seconds": initial.get("decode_seconds"),
            "max_decode_ahead": initial.get("max_decode_ahead"),
        }
        _dump(output / "metrics.json", metrics)
        if initial.get("state") != "completed" or initial.get("completed_images") != len(items):
            raise RuntimeError(f"cache run incomplete: {initial}")
        if resumed.get("decoded_images") or resumed.get("inference_batches"):
            raise RuntimeError(f"resume performed work: {resumed}")
        if failures:
            raise RuntimeError(f"cache has failed rows: {failures[:10]}")
        report.update(state="completed", initial=initial, resume=resumed,
                      strict_verify=verify, resume_verify=resume_verify,
                      status=status, failures=failures, metrics=metrics)
        _dump(report_path, report)
        print(json.dumps({"state": "completed", "images": len(items),
                          "cache_identity": identity, "output": str(output),
                          "initial": initial, "resume": resumed}, indent=2))
        return 0
    except Exception as exc:
        report.update(state="failed", error_type=type(exc).__name__, error_message=str(exc))
        _dump(report_path, report)
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
