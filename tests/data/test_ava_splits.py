from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_aesthetics.data.splits import (
    assign_cv,
    assign_outer,
    completed_records,
    connected_group_keys,
    load_partition,
    rating_deciles,
    sha256_json,
)


def records(count: int = 30) -> list[dict]:
    output = []
    for index in range(count):
        challenge = index // 5
        mean = 3.0 + (index % 10) * 0.3
        output.append({
            "image_id": str(index), "manifest_position": index, "challenge_id": challenge,
            "mean_rating": mean, "normalized_mean": (mean - 1) / 9,
            "vote_counts": [1] * 10, "vote_total": 10,
            "source_size": 10 + index, "source_path": str(Path(f"{index}.jpg")),
        })
    return output


def test_completed_membership_rejects_unknown_and_preserves_excluded_order():
    rows = records(6)
    complete, excluded = completed_records(rows, {"0", "2", "5"})
    assert [row["image_id"] for row in complete] == ["0", "2", "5"]
    assert excluded == ["1", "3", "4"]
    with pytest.raises(ValueError):
        completed_records(rows, {"missing"})


def test_outer_split_is_deterministic_group_disjoint_and_exactly_covers_rows():
    rows = records()
    keys = connected_group_keys(rows, [["0", "1"]])
    first = assign_outer(rows, keys, 20260920)
    second = assign_outer(rows, keys, 20260920)
    assert first == second
    assert set(first) == {row["image_id"] for row in rows}
    groups = {}
    for image_id, group in keys.items():
        groups.setdefault(group, set()).add(first[image_id])
    assert all(len(partitions) == 1 for partitions in groups.values())
    assert set(first.values()) == {"train", "validation", "test"}
    counts = {name: list(first.values()).count(name) for name in ("train", "validation", "test")}
    assert counts["train"] / len(rows) == pytest.approx(2 / 3, abs=1 / len(rows))
    assert sum(counts.values()) == len(rows)
    assert len(rating_deciles(rows)) == len(rows)
    assert sha256_json(rows) == sha256_json(rows)


def test_duplicate_edges_are_union_find_group_constraints():
    rows = records(10)
    # These IDs are in different challenge groups; the exact-duplicate edge
    # must still force one split assignment for both.
    keys = connected_group_keys(rows, [["0", "7"]])
    assert keys["0"] == keys["7"]
    outer = assign_outer(rows, keys, 20260920)
    assert outer["0"] == outer["7"]


def test_cv_is_group_disjoint_and_covers_development_only():
    rows = records(50)
    keys = connected_group_keys(rows, [["0", "1"]])
    outer = assign_outer(rows, keys, 20260920)
    development = [row for row in rows if outer[row["image_id"]] != "test"]
    cv = assign_cv(development, {row["image_id"]: keys[row["image_id"]] for row in development}, 5, 20260920)
    assert set(cv) == {row["image_id"] for row in development}
    assert set(cv.values()) == set(range(5))
    seen = {}
    for image_id, group in keys.items():
        if image_id in cv:
            seen.setdefault(group, set()).add(cv[image_id])
    assert all(len(folds) == 1 for folds in seen.values())
    assert not set(cv).intersection({row["image_id"] for row in rows if outer[row["image_id"]] == "test"})


def test_locked_test_loader_requires_explicit_opt_in(tmp_path: Path):
    path = tmp_path / "locked-test.jsonl"
    path.write_text(json.dumps({"image_id": "1"}) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        load_partition(path, "test")
    assert load_partition(path, "test", allow_locked_test=True) == [{"image_id": "1"}]


def test_partition_rejects_unknown_name(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"image_id": "1"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_partition(path, "unknown", allow_locked_test=True)
