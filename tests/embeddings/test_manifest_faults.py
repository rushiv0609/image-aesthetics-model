from __future__ import annotations

import json
import hashlib
from pathlib import Path
import pytest

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


def test_manifest_recognizes_materializer_source_sha256_and_rejects_conflicts(tmp_path):
    source=tmp_path/"image.jpg"; source.write_bytes(b"fixture")
    digest=hashlib.sha256(b"fixture").hexdigest()
    path=tmp_path/"manifest.jsonl"
    path.write_text(json.dumps({"image_id":"image","source_path":"image.jpg","source_sha256":digest})+"\n")
    assert load_manifest(path)[0].source_fingerprint==digest
    path.write_text(json.dumps({"image_id":"image","source_path":"image.jpg","sha256":"0"*64,"source_sha256":"1"*64})+"\n")
    with pytest.raises(ValueError,match="conflicting source fingerprint"):
        load_manifest(path)


def test_rooted_manifest_rejects_traversal_absolute_and_symlink_and_preserves_nested_paths(tmp_path):
    root=tmp_path/"images"; (root/"nested").mkdir(parents=True)
    source=root/"nested"/"a.jpg"; source.write_bytes(b"a")
    manifest=tmp_path/"manifest.jsonl"
    manifest.write_text(json.dumps({"image_id":"a","source_path":"nested/a.jpg"})+"\n")
    assert load_manifest(manifest,image_root=root)[0].source_path==str(source.resolve())
    for name in ("../outside.jpg",str(source)):
        manifest.write_text(json.dumps({"image_id":"a","source_path":name})+"\n")
        with pytest.raises(ValueError,match="safe relative"):
            load_manifest(manifest,image_root=root)
    outside=tmp_path/"outside.jpg"; outside.write_bytes(b"x")
    (root/"link.jpg").symlink_to(outside)
    manifest.write_text(json.dumps({"image_id":"link","source_path":"link.jpg"})+"\n")
    with pytest.raises(ValueError,match="symlink"):
        load_manifest(manifest,image_root=root)


def test_preprocessor_has_one_frozen_pillow_backend(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image
    import image_aesthetics.embeddings.preprocess as prep
    path=tmp_path/"image.png"; Image.new("RGB",(9,5),(20,80,120)).save(path)
    monkeypatch.setattr(prep,"_prepare_with_macos_imageio",lambda *_: (_ for _ in ()).throw(AssertionError("alternate decoder called")))
    pixels,decoder=prep.prepare_image(path,16)
    assert decoder=="pillow" and pixels.shape==(16,16,3) and pixels.dtype==np.uint8
    assert prep.preprocessing_contract()["decoder"]=="pillow"
