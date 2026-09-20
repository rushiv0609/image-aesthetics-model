import json
import stat
from zipfile import ZipFile, ZipInfo

import pytest

from image_aesthetics.data.acquisition import (
    safe_member_path,
    extract_zip_safely,
    verify,
    write_manifest,
    write_multi_manifest,
)


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
    assert verify(tmp_path) == (True, "verified 1 file(s)")
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


def test_multi_file_manifest_is_deterministic_and_verifiable(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    first = downloads / "part-1.zip"
    second = downloads / "part-2.zip"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    manifest = write_multi_manifest(
        tmp_path,
        files=[
            (second, "https://example.test/part-2.zip"),
            (first, "https://example.test/part-1.zip"),
        ],
        source="research-mirror",
    )
    payload = json.loads(manifest.read_text())
    assert payload["manifest_version"] == 2
    assert [item["filename"] for item in payload["files"]] == [
        "downloads/part-1.zip",
        "downloads/part-2.zip",
    ]
    assert verify(tmp_path) == (True, "verified 2 file(s)")
    second.write_bytes(b"corrupt")
    assert verify(tmp_path) == (False, "size mismatch: downloads/part-2.zip")


def test_multi_file_manifest_rejects_external_and_duplicate_files(tmp_path):
    inside = tmp_path / "inside.zip"
    inside.write_bytes(b"inside")
    outside = tmp_path.parent / "outside-multi.zip"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="outside dataset root"):
        write_multi_manifest(tmp_path, files=[(outside, "https://example.test/out")], source="test")
    with pytest.raises(ValueError, match="duplicate"):
        write_multi_manifest(
            tmp_path,
            files=[(inside, "https://example.test/a"), (inside, "https://example.test/b")],
            source="test",
        )


def test_extract_zip_safely_is_atomic_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as source:
        source.writestr("nested/file.txt", "payload")
    destination = tmp_path / "raw"
    assert extract_zip_safely(archive, destination) == 1
    assert (destination / "nested/file.txt").read_text() == "payload"
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_zip_safely(archive, destination)


def test_extract_zip_safely_rejects_traversal_and_symlinks(tmp_path):
    traversal = tmp_path / "traversal.zip"
    with ZipFile(traversal, "w") as source:
        source.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="escapes"):
        extract_zip_safely(traversal, tmp_path / "raw-traversal")

    symlink = tmp_path / "symlink.zip"
    member = ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(symlink, "w") as source:
        source.writestr(member, "target")
    with pytest.raises(ValueError, match="symlink"):
        extract_zip_safely(symlink, tmp_path / "raw-symlink")
