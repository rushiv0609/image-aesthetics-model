import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from PIL import Image
from image_aesthetics.data.adapters import TAD66KAdapter
from image_aesthetics.data.preparation import prepare_dataset

def test_prepare_writes_metadata_rejections_and_hash_without_absolute_paths(tmp_path: Path):
    labels = tmp_path / "labels.zip"
    with ZipFile(labels, "w", ZIP_DEFLATED) as archive:
        archive.writestr("labels/merge/train.csv", "image,score\na.jpg,5\nmissing.jpg,6\n")
        archive.writestr("labels/merge/test.csv", "image,score\n")
    image_root = tmp_path / "images"; image_root.mkdir()
    Image.new("RGB", (13, 7), (1, 2, 3)).save(image_root / "a.jpg")
    output = tmp_path / "prepared"
    audit = prepare_dataset(TAD66KAdapter(labels), image_root=image_root, output_root=output, source_files=(labels,))
    assert audit["accepted"] == 1 and audit["rejected"] == 1
    sample = json.loads((output / "samples.jsonl").read_text())
    assert sample["width"] == 13 and sample["height"] == 7 and sample["decode_status"] == "ok"
    assert len(sample["sha256"]) == 64
    assert str(tmp_path) not in (output / "samples.jsonl").read_text()
    rejected = json.loads((output / "rejected.jsonl").read_text())
    assert rejected["reason"] == "missing_image"

def test_prepare_is_immutable_and_byte_deterministic(tmp_path: Path):
    labels = tmp_path / "labels.zip"
    with ZipFile(labels, "w", ZIP_DEFLATED) as archive:
        archive.writestr("labels/merge/train.csv", "image,score\na.jpg,5\n")
        archive.writestr("labels/merge/test.csv", "image,score\n")
    image_root = tmp_path / "images"; image_root.mkdir()
    Image.new("RGB", (3, 2), (4, 5, 6)).save(image_root / "a.jpg")
    first = tmp_path / "one"; second = tmp_path / "two"
    prepare_dataset(TAD66KAdapter(labels), image_root=image_root, output_root=first, source_files=(labels,))
    prepare_dataset(TAD66KAdapter(labels), image_root=image_root, output_root=second, source_files=(labels,))
    for name in ("samples.jsonl", "rejected.jsonl", "audit.json", "source-manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    try:
        prepare_dataset(TAD66KAdapter(labels), image_root=image_root, output_root=first, source_files=(labels,))
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing prepared snapshots must be immutable")
