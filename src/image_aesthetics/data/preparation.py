"""Deterministic, dependency-light preparation of adapter samples.

This layer resolves adapter image names against an externally managed image root,
records decode metadata, and writes durable JSONL manifests. It never downloads,
extracts, or modifies source data.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from image_aesthetics.data.adapters import CanonicalSample, DatasetAdapter

SCHEMA_VERSION = "canonical-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_jsonl(path, [payload])


def _index_images(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        index.setdefault(path.name, []).append(path)
    return index


def _resolve(sample: CanonicalSample, root: Path, index: dict[str, list[Path]]) -> tuple[Path | None, str | None]:
    direct = root / sample.image_name
    if direct.is_file():
        return direct, None
    candidates = index.get(Path(sample.image_name).name, [])
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "missing_image"
    return None, "ambiguous_image_name"


def prepare_dataset(adapter: DatasetAdapter, *, image_root: Path, output_root: Path, source_files: Iterable[Path] = ()) -> dict:
    """Materialize one adapter into samples/rejected/audit/source-manifest.

    Paths are intentionally absent from sample records; only relative source names,
    hashes, and structural metadata are persisted.
    """
    image_root = Path(image_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"prepared snapshot already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    index = _index_images(image_root)
    accepted: list[dict] = []
    rejected: list[dict] = []
    for item in adapter.iter_samples(include_rejections=True):
        if not isinstance(item, CanonicalSample):
            rejected.append({"dataset_id": adapter.dataset_id, "image_name": item[0], "reason": item[1]})
            continue
        path, reason = _resolve(item, image_root, index)
        if reason:
            rejected.append({"dataset_id": item.dataset_id, "image_id": item.image_id, "image_name": item.image_name, "reason": reason})
            continue
        try:
            digest = _sha256(path)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image_format = image.format
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            rejected.append({"dataset_id": item.dataset_id, "image_id": item.image_id, "image_name": item.image_name, "reason": "decode_failed", "error_type": type(exc).__name__})
            continue
        record = asdict(item)
        record.update({"schema_version": SCHEMA_VERSION, "sha256": digest, "width": width, "height": height, "format": image_format, "decode_status": "ok"})
        record["native_score_range"] = list(item.native_score_range)
        accepted.append(record)
    accepted.sort(key=lambda row: (row["dataset_id"], row["image_id"]))
    rejected.sort(key=lambda row: (row.get("image_name", ""), row["reason"]))
    try:
        _atomic_jsonl(staging / "samples.jsonl", accepted)
        _atomic_jsonl(staging / "rejected.jsonl", rejected)
        source_manifest = {"schema_version": SCHEMA_VERSION, "dataset_id": adapter.dataset_id, "annotation_files": []}
        for source in sorted((Path(value) for value in source_files), key=lambda value: value.name):
            if source.exists() and source.is_file():
                source_manifest["annotation_files"].append({"name": source.name, "sha256": _sha256(source), "size": source.stat().st_size})
        audit = {"schema_version": SCHEMA_VERSION, "dataset_id": adapter.dataset_id, "accepted": len(accepted), "rejected": len(rejected), "rejection_reasons": {reason: sum(1 for row in rejected if row["reason"] == reason) for reason in sorted({row["reason"] for row in rejected})}, "source_manifest": "source-manifest.json"}
        _atomic_json(staging / "audit.json", audit)
        _atomic_json(staging / "source-manifest.json", source_manifest)
        os.replace(staging, output_root)
        return audit
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
