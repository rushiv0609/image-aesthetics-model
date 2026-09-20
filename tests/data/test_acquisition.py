import json

import pytest

from image_aesthetics.data.acquisition import safe_member_path, verify, write_manifest


def test_safe_member_path_rejects_escape(tmp_path):
    assert safe_member_path(tmp_path, "downloads/source.zip") == (tmp_path / "downloads/source.zip").resolve()
    with pytest.raises(ValueError, match="escapes"):
        safe_member_path(tmp_path, "../outside.zip")
    with pytest.raises(ValueError, match="escapes"):
        safe_member_path(tmp_path, "/tmp/outside.zip")


def test_manifest_is_immutable_and_verifiable(tmp_path):
    source = tmp_path / "downloads/source.zip"
    source.parent.mkdir()
    source.write_bytes(b"dataset")
    manifest = write_manifest(
        tmp_path, url="https://example.test/source.zip", source="user-supplied",
        path=source, user_supplied=True,
    )
    assert json.loads(manifest.read_text())["redistribution"] == "prohibited"
    assert verify(tmp_path) == (True, "verified")
    source.write_bytes(b"changed")
    assert verify(tmp_path)[0] is False
    with pytest.raises(FileExistsError, match="immutable"):
        write_manifest(
            tmp_path, url="https://example.test/source.zip", source="user-supplied",
            path=source, user_supplied=True,
        )


def test_verify_rejects_manifest_path_escape(tmp_path):
    outside = tmp_path.parent / "outside.bin"
    outside.write_bytes(b"outside")
    (tmp_path / "download-manifest.json").write_text(
        json.dumps({"filename": "../outside.bin", "size": 7, "sha256": "unused"})
    )
    assert verify(tmp_path) == (False, "invalid manifest")
