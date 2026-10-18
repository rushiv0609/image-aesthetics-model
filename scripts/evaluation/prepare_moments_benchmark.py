#!/usr/bin/env python3
"""Prepare an external, privacy-safe event-grouped Moments rating manifest."""
from __future__ import annotations

import argparse, hashlib, hmac, json, os
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi"}


def _timestamp(path: Path) -> float:
    try:
        from PIL import Image, ExifTags
        with Image.open(path) as image:
            exif = image.getexif()
            names = {value: key for key, value in ExifTags.TAGS.items()}
            value = exif.get(names.get("DateTimeOriginal")) or exif.get(names.get("DateTime"))
            if value:
                return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S").timestamp()
    except Exception:
        pass
    return path.stat().st_mtime


def prepare(media_root: Path, salt: bytes, *, event_gap_hours: float = 6.0) -> list[dict]:
    media_root = media_root.resolve()
    paths = sorted(path for path in media_root.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)
    rows = []
    for path in paths:
        relative = path.relative_to(media_root).as_posix()
        opaque = hmac.new(salt, relative.encode(), hashlib.sha256).hexdigest()
        rows.append({"path": path, "relative": relative, "item_id": opaque,
                     "timestamp": _timestamp(path),
                     "media_type": "video" if path.suffix.lower() in VIDEO_EXTENSIONS else "image"})
    rows.sort(key=lambda row: (row["timestamp"], row["relative"]))
    event_index, previous = -1, None
    gap = event_gap_hours * 3600
    for row in rows:
        if previous is None or row["timestamp"] - previous > gap:
            event_index += 1
        previous = row["timestamp"]
        row["event_id"] = hmac.new(salt, f"event:{event_index}".encode(), hashlib.sha256).hexdigest()
    return [{"schema_version": "moments-benchmark-item-v1", "item_id": row["item_id"],
             "event_id": row["event_id"], "media_type": row["media_type"],
             # Paths remain only in the external private manifest.
             "media_path": str(row["path"]), "capture_time_unix": row["timestamp"]}
            for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-gap-hours", type=float, default=6.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    salt_path = args.output_dir / "id-salt.bin"
    if salt_path.exists():
        salt = salt_path.read_bytes()
    else:
        salt = os.urandom(32); salt_path.write_bytes(salt); salt_path.chmod(0o600)
    rows = prepare(args.media_root, salt, event_gap_hours=args.event_gap_hours)
    manifest = args.output_dir / "manifest.private.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    annotations = args.output_dir / "annotations.jsonl"
    annotations.touch(exist_ok=True)
    summary = {"schema_version": "moments-benchmark-preparation-v1", "items": len(rows),
               "events": len({row["event_id"] for row in rows}),
               "images": sum(row["media_type"] == "image" for row in rows),
               "videos": sum(row["media_type"] == "video" for row in rows),
               "human_annotations_present": annotations.stat().st_size > 0,
               "private_media_paths_committed": False}
    (args.output_dir / "preparation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
