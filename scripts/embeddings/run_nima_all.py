#!/usr/bin/env python3
"""Sequentially generate the pretrained NIMA cache for all prepared datasets."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
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


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--datasets", nargs="*", choices=[row[0] for row in DATASETS])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-workers", type=int, default=8)
    args = parser.parse_args()
    selected = set(args.datasets or [row[0] for row in DATASETS])
    rows = [row for row in DATASETS if row[0] in selected]
    status_dir = args.base_dir / "fusion" / "features" / "nima-v1"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "orchestrator-status.json"
    status = {"state": "running", "started_at": now(), "updated_at": now(),
              "dataset_order": [row[0] for row in rows], "completed_datasets": [],
              "current_dataset": None, "signal": "nima", "batch_size": args.batch_size}
    write(status_path, status)
    for dataset, manifest_version, raw_dir in rows:
        root = args.base_dir / dataset
        manifest = root / "prepared" / "feature-manifests" / manifest_version / "features.jsonl"
        image_root = root / raw_dir
        output = root / "features" / "nima-v1"
        output.mkdir(parents=True, exist_ok=True)
        status.update(current_dataset=dataset, updated_at=now())
        write(status_path, status)
        command = [sys.executable, str(ROOT / "scripts/embeddings/cache_nima.py"),
                   "--manifest", str(manifest), "--image-root", str(image_root),
                   "--output", str(output), "--batch-size", str(args.batch_size),
                   "--decode-workers", str(args.decode_workers)]
        with (output / "orchestrator.log").open("a") as log:
            log.write(f"[{now()}] START {dataset}: {command}\n")
            log.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            log.write(f"[{now()}] END {dataset}: exit={result.returncode}\n")
        if result.returncode:
            status.update(state="failed", failed_dataset=dataset, updated_at=now())
            write(status_path, status)
            return result.returncode
        status["completed_datasets"].append(dataset)
        status.update(current_dataset=None, updated_at=now())
        write(status_path, status)
    status.update(state="completed", current_dataset=None, updated_at=now(), finished_at=now())
    write(status_path, status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
