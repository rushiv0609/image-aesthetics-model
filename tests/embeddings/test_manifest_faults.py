from __future__ import annotations

import json
from pathlib import Path

from image_aesthetics.embeddings.preprocess import load_manifest


def test_manifest_keeps_source_identity_when_file_is_missing(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "image_id": "missing", "source_path": "gone.jpg",
        "source_size": 12, "source_mtime_ns": 34,
    }) + "\n", encoding="utf-8")
    item = load_manifest(manifest)[0]
    assert item.source_size == 12
    assert item.source_mtime_ns == 34
    assert item.source_path == str(tmp_path / "gone.jpg")
