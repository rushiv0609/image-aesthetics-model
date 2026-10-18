#!/usr/bin/env python3
"""Materialize one deterministic, image-like feature source per Moments item.

Images already decodable by the cache pipeline are hard-linked without pixel
changes. HEIC inputs are converted to quality-95 JPEG with macOS ``sips``.
Videos are represented by their temporal midpoint, decoded by pinned command
arguments through ffmpeg. The output lives outside Git and is immutable once
its provenance report is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


SCHEMA = "moments-feature-media-v1"
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = {
    "images": "hard-link original JPEG/PNG/TIFF/WebP bytes",
    "heic": "sips -s format jpeg -s formatOptions 95",
    "videos": "ffmpeg temporal midpoint, one JPEG frame, -q:v 2",
    "video_aggregation": "one deterministic midpoint representative per file",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _duration(path: Path, runner=_run) -> float:
    result = runner([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    duration = float(result.stdout.strip())
    if not 0 < duration < 24 * 3600:
        raise ValueError(f"invalid video duration for {path}: {duration}")
    return duration


def _materialize_one(row: dict, destination: Path, runner=_run) -> dict:
    source = Path(row["media_path"]).resolve(strict=True)
    kind = str(row["media_type"])
    if kind == "video":
        duration = _duration(source, runner)
        offset = duration / 2.0
        runner(["ffmpeg", "-v", "error", "-ss", f"{offset:.6f}", "-i", str(source),
                "-frames:v", "1", "-q:v", "2", str(destination)])
        method = "video-midpoint-ffmpeg"
    elif source.suffix.lower() == ".heic":
        runner(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "95",
                str(source), "--out", str(destination)])
        offset = None
        method = "heic-sips-jpeg95"
    else:
        os.link(source, destination)
        offset = None
        method = "hard-link-original"
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"feature media was not created: {destination}")
    return {
        "schema_version": SCHEMA,
        "image_id": str(row["item_id"]),
        "source_path": destination.name,
        "source_sha256": sha256_file(destination),
        "source_size": destination.stat().st_size,
        "source_mtime_ns": destination.stat().st_mtime_ns,
        "benchmark_media_type": kind,
        "materialization_method": method,
        "video_offset_seconds": offset,
        "eligible_for_fusion": False,
        "eligible_as_aesthetic_target": False,
        "training_role": "private-product-benchmark-only",
        "locked_test": False,
    }


def materialize(manifest: Path, output_dir: Path, *, runner=_run) -> dict:
    manifest = Path(manifest).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("private benchmark feature media must remain outside Git")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    media_dir = output_dir / "media"
    media_dir.mkdir()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or len({str(row["item_id"]) for row in rows}) != len(rows):
        raise ValueError("benchmark manifest is empty or has duplicate item IDs")
    derived = []
    for row in rows:
        suffix = Path(row["media_path"]).suffix.lower()
        destination_suffix = ".jpg" if row["media_type"] == "video" or suffix == ".heic" else suffix
        derived.append(_materialize_one(row, media_dir / f"{row['item_id']}{destination_suffix}", runner))
    feature_manifest = output_dir / "features.jsonl"
    feature_manifest.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in derived))
    report = {
        "schema_version": SCHEMA,
        "state": "completed",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "feature_manifest": str(feature_manifest),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "media_root": str(media_dir),
        "items": len(derived),
        "images": sum(row["benchmark_media_type"] == "image" for row in derived),
        "videos": sum(row["benchmark_media_type"] == "video" for row in derived),
        "methods": {method: sum(row["materialization_method"] == method for row in derived)
                    for method in sorted({row["materialization_method"] for row in derived})},
        "preprocessing_contract": CONTRACT,
        "human_labels_accessed": False,
        "locked_test_accessed": False,
    }
    (output_dir / "materialization-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        result = materialize(args.manifest, args.output_dir)
    else:
        rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
        result = {"schema_version": SCHEMA, "state": "planned", "items": len(rows),
                  "output_dir": str(args.output_dir), "preprocessing_contract": CONTRACT,
                  "locked_test_accessed": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
