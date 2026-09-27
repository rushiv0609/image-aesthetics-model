#!/usr/bin/env python3
"""Run the remaining-four cache sequentially for all prepared datasets."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/Users/rushivyas/datasets/image-aesthetics")
DATASETS = (
    ("aadb", "aadb-v1", "raw/v1"),
    ("koniq10k", "koniq10k-v1", "raw/v1"),
    ("spaq", "spaq-v1", "raw/v1/TestImage"),
    ("para", "para-v1", "raw/v1"),
    ("flickr_aes", "flickr-aes-v1", "raw/v1"),
    ("tad66k", "tad66k-v1", "raw/v1"),
    ("ava", "ava-v2", "raw/images/ava_images"),
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verify_local_cache(cache: Path, manifest: Path, image_root: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/embeddings/cache_ten_signal.py"),
         "verify", "--cache-dir", str(cache), "--manifest", str(manifest),
         "--image-root", str(image_root)],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"local-five verification returned no JSON: {result.stderr}") from None
    if result.returncode or value.get("errors"):
        raise RuntimeError(f"local-five cache is not complete and verified: {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", choices=[d[0] for d in DATASETS])
    parser.add_argument("--base-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--chunk-images", type=int, default=8000,
                        help="restart the model process after this many new rows to bound process RSS")
    parser.add_argument("--max-attempts-per-dataset", type=int, default=3)
    args = parser.parse_args(argv)
    if args.chunk_images < 1:
        raise SystemExit("--chunk-images must be positive")
    selected = set(args.datasets or [row[0] for row in DATASETS])
    datasets = [row for row in DATASETS if row[0] in selected]
    base = args.base_dir.resolve(strict=True)
    orchestration = base / "fusion" / "features" / "remaining-four-v1"
    orchestration.mkdir(parents=True, exist_ok=True)
    status_path = orchestration / "orchestrator-status.json"
    status = {
        "schema_version": 1,
        "state": "running",
        "started_at": _now(),
        "updated_at": _now(),
        "current_dataset": None,
        "completed_datasets": [],
        "failed_dataset": None,
        "dataset_order": [row[0] for row in datasets],
        "signals": ["clip_embedding", "laion_clip", "musiq_aesthetic", "musiq_technical"],
        "blocked_signal": "nima",
        "device": "mps if available, otherwise CPU",
    }
    _write(status_path, status)

    for dataset_id, manifest_version, raw_dir in datasets:
        dataset_root = base / dataset_id
        manifest = dataset_root / "prepared" / "feature-manifests" / manifest_version / "features.jsonl"
        image_root = dataset_root / raw_dir
        local_cache = dataset_root / "features" / ("local-five-v2" if dataset_id == "ava" else "local-five-v1") / "cache"
        remaining = dataset_root / "features" / "remaining-four-v1"
        join_dir = dataset_root / "features" / "joined-nine-v1"
        if not manifest.is_file() or not image_root.is_dir() or not (local_cache / "work.sqlite3").is_file():
            status.update(state="blocked", failed_dataset=dataset_id, updated_at=_now(),
                          error=f"missing manifest, image root, or local-five cache for {dataset_id}")
            _write(status_path, status)
            return 2

        try:
            local_verify = _verify_local_cache(local_cache, manifest, image_root)
        except Exception as exc:
            status.update(state="blocked", failed_dataset=dataset_id, updated_at=_now(),
                          error=f"local-five preflight failed: {exc}")
            _write(status_path, status)
            return 2

        status.update(current_dataset=dataset_id, updated_at=_now(), failed_dataset=None,
                      local_five_preflight=local_verify)
        _write(status_path, status)
        remaining.mkdir(parents=True, exist_ok=True)
        log_path = remaining / "orchestrator.log"
        command = [
            sys.executable, str(ROOT / "scripts/embeddings/cache_remaining_four.py"),
            "--manifest", str(manifest), "--image-root", str(image_root),
            "--output", str(remaining), "--batch-size", str(args.batch_size),
            "--shard-size", str(args.shard_size), "--max-new-images", str(args.chunk_images),
        ]
        return_code = 0
        attempt = 1
        chunk_number = 0
        while True:
            status.update(attempt=attempt, chunk_number=chunk_number + 1, updated_at=_now())
            _write(status_path, status)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{_now()}] START {dataset_id} attempt {attempt} chunk {chunk_number + 1}: {command}\n")
                log.flush()
                started = time.monotonic()
                child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
                elapsed = time.monotonic() - started
                log.write(f"[{_now()}] END {dataset_id} attempt {attempt} chunk {chunk_number + 1}: exit={child.returncode} elapsed={elapsed:.1f}s\n")
                return_code = child.returncode
            if return_code:
                if attempt >= args.max_attempts_per_dataset:
                    break
                attempt += 1
                continue
            try:
                progress = json.loads((remaining / "cache" / "progress.json").read_text())
            except (OSError, json.JSONDecodeError) as exc:
                return_code = 2
                status.update(state="failed", failed_dataset=dataset_id, updated_at=_now(),
                              error=f"cache chunk produced no valid progress record: {exc}")
                _write(status_path, status)
                return return_code
            if progress.get("state") == "completed":
                break
            if progress.get("state") == "incomplete" and not progress.get("failed_images"):
                chunk_number += 1
                continue
            if attempt < args.max_attempts_per_dataset:
                attempt += 1
                continue
            return_code = 2
            break
        if return_code:
            status.update(state="failed", failed_dataset=dataset_id, updated_at=_now(),
                          error=f"cache chunk/retry loop exited {return_code} after {args.max_attempts_per_dataset} attempts")
            _write(status_path, status)
            return return_code

        join_command = [
            sys.executable, str(ROOT / "scripts/embeddings/join_feature_caches.py"),
            "--local-cache", str(local_cache), "--remaining-cache", str(remaining / "cache"),
            "--manifest", str(manifest), "--image-root", str(image_root),
            "--dataset-id", dataset_id, "--output", str(join_dir),
        ]
        join_log = remaining / "join.log"
        with join_log.open("a", encoding="utf-8") as log:
            join = subprocess.run(join_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if join.returncode:
            status.update(state="failed", failed_dataset=dataset_id, updated_at=_now(),
                          error=f"identity-checked feature join exited {join.returncode}")
            _write(status_path, status)
            return join.returncode

        status["completed_datasets"].append(dataset_id)
        status.update(current_dataset=None, updated_at=_now())
        _write(status_path, status)

    status.update(state="completed", current_dataset=None, updated_at=_now(), finished_at=_now())
    _write(status_path, status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
