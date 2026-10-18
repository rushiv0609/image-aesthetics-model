import json

import pytest

from scripts.evaluation.filter_moments_unreadable import filter_unreadable


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_filters_only_decode_failures_before_annotations(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"; features = tmp_path / "features.jsonl"
    annotations = tmp_path / "annotations.jsonl"; annotations.touch()
    failures = tmp_path / "failures.json"
    _jsonl(benchmark, [{"item_id": "a"}, {"item_id": "b"}])
    _jsonl(features, [{"image_id": "a"}, {"image_id": "b"}])
    failures.write_text(json.dumps([{"image_id": "b", "error_type": "UnidentifiedImageError",
                                     "error_message": "bad bytes"}]))
    report = filter_unreadable(benchmark, features, annotations, failures, tmp_path / "out")
    assert report["valid_items"] == 1 and report["excluded_items"] == 1
    assert json.loads((tmp_path / "out/manifest.private.validated.jsonl").read_text())["item_id"] == "a"


def test_refuses_filter_after_human_annotations(tmp_path):
    annotations = tmp_path / "annotations.jsonl"; annotations.write_text("{}\n")
    for name in ("benchmark.jsonl", "features.jsonl"):
        (tmp_path / name).write_text("{}\n")
    failures = tmp_path / "failures.json"; failures.write_text("[]")
    with pytest.raises(ValueError, match="after human annotation"):
        filter_unreadable(tmp_path / "benchmark.jsonl", tmp_path / "features.jsonl",
                          annotations, failures, tmp_path / "out")
