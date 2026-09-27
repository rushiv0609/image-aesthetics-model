import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from image_aesthetics.data.manifest_materialization import materialize_feature_manifest


def _image_bytes(image_format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (8, 6), (30, 90, 180)).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("dataset", ["ava", "tad66k", "spaq", "para", "flickr_aes", "koniq10k", "aadb"])
def test_materializes_prepared_layouts(tmp_path, dataset):
    prepared = tmp_path / "prepared"; prepared.mkdir()
    images = tmp_path / "images"; images.mkdir()
    image_path = images / "nested" / "one.jpg"; image_path.parent.mkdir(); image_path.write_bytes(_image_bytes())
    sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    if dataset == "ava":
        row = {"image_id": "1", "source_path": str(image_path), "source_size": image_path.stat().st_size,
               "source_mtime_ns": image_path.stat().st_mtime_ns, "mean_rating": 6.0}
        (prepared / "dataset.jsonl").write_text(json.dumps(row) + "\n")
    elif dataset == "aadb":
        row = {"identifier": "one.jpg", "path": "nested/one.jpg", "score": 0.7, "sha256": sha, "split": "train"}
        (prepared / "images.jsonl").write_text(json.dumps({"identifier": "one.jpg", "path": "nested/one.jpg", "sha256": sha}) + "\n")
        (prepared / "samples.jsonl").write_text(json.dumps(row) + "\n")
    else:
        name = "imgs/nested/one.jpg" if dataset == "para" else "nested/one.jpg"
        if dataset in ("spaq", "koniq10k", "flickr_aes"):
            name = "one.jpg"
        row = {"dataset_id": dataset, "image_id": "one.jpg", "image_name": name,
               "sha256": sha, "native_score": 4.0, "split": "train"}
        if dataset == "flickr_aes":
            (prepared / "images.jsonl").write_text(json.dumps({"image_id": "one.jpg", "image_name": "nested/one.jpg", "sha256": sha}) + "\n")
            (prepared / "samples.jsonl").write_text(json.dumps(row) + "\n" + json.dumps({**row, "metadata": {"worker_id": "w2"}}) + "\n")
        else:
            (prepared / "samples.jsonl").write_text(json.dumps(row) + "\n")
    output = tmp_path / "out"
    audit = materialize_feature_manifest(dataset, prepared, images, output)
    output_repeat = tmp_path / "out-repeat"
    repeated_audit = materialize_feature_manifest(dataset, prepared, images, output_repeat)
    assert audit == repeated_audit
    for name in ("features.jsonl", "labels.jsonl", "rejected.jsonl", "alias-split-conflicts.jsonl", "audit.json"):
        assert (output/name).read_bytes() == (output_repeat/name).read_bytes()
    features = [json.loads(line) for line in (output / "features.jsonl").read_text().splitlines()]
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    assert audit["feature_rows"] == 1
    assert features[0]["image_id"].startswith(dataset + ":")
    assert features[0]["source_path"] == "nested/one.jpg"
    assert labels and all(row["join_feature_id"] == features[0]["image_id"] for row in labels)
    if dataset == "flickr_aes":
        assert audit["label_rows"] == 2
    if dataset == "spaq":
        assert features[0]["label_task"] == "technical"
        assert features[0]["training_role"] == "technical_auxiliary"
        assert not features[0]["eligible_as_aesthetic_target"]
        assert all(label["label_task"] == "technical" and not label["eligible_as_aesthetic_target"] for label in labels)
    if dataset == "koniq10k":
        assert features[0]["training_role"] == "technical_diagnostic_only"
        assert not features[0]["eligible_for_fusion"] and not features[0]["eligible_as_aesthetic_target"]


def test_materializer_rejects_traversal_and_symlinks(tmp_path):
    prepared = tmp_path / "prepared"; prepared.mkdir()
    images = tmp_path / "images"; images.mkdir()
    outside = tmp_path / "outside.jpg"; outside.write_bytes(_image_bytes())
    (images / "linked.jpg").symlink_to(outside)
    rows = [
        {"dataset_id": "tad66k", "image_id": "escape", "image_name": "../outside.jpg", "native_score": 1},
        {"dataset_id": "tad66k", "image_id": "link", "image_name": "linked.jpg", "native_score": 1},
    ]
    (prepared / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = materialize_feature_manifest("tad66k", prepared, images, tmp_path / "out")
    assert audit["feature_rows"] == 0 and audit["rejected_source_rows"] == 2


def test_materializer_quarantines_ambiguous_basename(tmp_path):
    prepared = tmp_path / "prepared"; prepared.mkdir()
    images = tmp_path / "images"; (images / "a").mkdir(parents=True); (images / "b").mkdir()
    (images / "a" / "same.jpg").write_bytes(_image_bytes())
    (images / "b" / "same.jpg").write_bytes(_image_bytes("JPEG"))
    row = {"dataset_id": "spaq", "image_id": "same.jpg", "image_name": "same.jpg", "native_score": 1}
    (prepared / "samples.jsonl").write_text(json.dumps(row) + "\n")
    audit = materialize_feature_manifest("spaq", prepared, images, tmp_path / "out")
    assert audit["rejected_source_rows"] == 1


def test_exact_aliases_crossing_native_splits_are_quarantined_and_audited(tmp_path):
    prepared=tmp_path/"prepared"; prepared.mkdir()
    images=tmp_path/"images"; images.mkdir()
    identical_image = _image_bytes()
    (images/"train.jpg").write_bytes(identical_image)
    (images/"test.jpg").write_bytes(identical_image)
    rows=[{"image_id":"train","image_name":"train.jpg","score":.8,"split":"train"},
          {"image_id":"test","image_name":"test.jpg","score":.2,"split":"test"}]
    (prepared/"images.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    (prepared/"samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    out=tmp_path/"out"
    audit=materialize_feature_manifest("aadb",prepared,images,out)
    feature=json.loads((out/"features.jsonl").read_text())
    labels=[json.loads(line) for line in (out/"labels.jsonl").read_text().splitlines()]
    conflicts=[json.loads(line) for line in (out/"alias-split-conflicts.jsonl").read_text().splitlines()]
    assert audit["feature_rows"]==1 and audit["exact_content_alias_groups"]==1
    assert audit["cross_native_split_alias_quarantine_groups"]==1
    assert feature["quarantine"] and feature["quarantine_reason"]=="exact_byte_aliases_span_native_splits"
    assert len(conflicts)==1 and conflicts[0]["native_splits"]==["test","train"]
    assert len(labels)==2 and all(label["quarantine"] for label in labels)


@pytest.mark.parametrize("image_bytes", [b"not an image", _image_bytes("JPEG")[:-24]])
def test_materializer_rejects_undecodable_and_truncated_images_with_durable_reason(tmp_path, image_bytes):
    prepared = tmp_path / "prepared"; prepared.mkdir()
    images = tmp_path / "images"; images.mkdir()
    image_path = images / "bad.jpg"; image_path.write_bytes(image_bytes)
    row = {"image_id": "bad", "image_name": "bad.jpg", "score": 0.5}
    (prepared / "samples.jsonl").write_text(json.dumps(row) + "\n")

    audit = materialize_feature_manifest("tad66k", prepared, images, tmp_path / "out")
    rejected = [json.loads(line) for line in (tmp_path / "out" / "rejected.jsonl").read_text().splitlines()]
    assert audit["schema_version"] == "feature-manifest-v2"
    assert audit["feature_rows"] == 0 and audit["rejected_source_rows"] == 1
    assert rejected[0]["reason"].startswith("source_decode_invalid: Pillow decode failed")
