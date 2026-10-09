from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from image_aesthetics.data.fusion_training_dataset import (
    build_ranking_pairs,
    prepare_spaq_technical_splits,
)
from image_aesthetics.data.normalization import FeatureNormalizer, load_stats


def test_streaming_stats_are_atomic_and_identity_bound(tmp_path: Path):
    stats_path = tmp_path / "fold-2.json"
    identity = {"manifest_sha256": "abc", "partition": "train", "validation_fold": 2}
    normalizer = FeatureNormalizer.fit(
        (np.asarray(row, dtype=np.float32) for row in ([1, 2], [3, 4], [5, 6])),
        identity=identity, stats_path=stats_path,
    )
    assert stats_path.is_file()
    assert np.allclose(normalizer.stats.mean, [3, 4])
    assert np.allclose(normalizer.stats.std, np.sqrt([8 / 3, 8 / 3]))
    loaded = load_stats(stats_path, expected_identity=identity)
    assert loaded.stats_hash == normalizer.stats.stats_hash
    with pytest.raises(ValueError, match="identity"):
        load_stats(stats_path, expected_identity={**identity, "validation_fold": 1})
    assert not list(tmp_path.glob(".*fold-2.json.*"))


def test_ranking_pairs_are_same_dataset_and_tad_theme_only():
    rows = [
        {"dataset_id": "tad66k", "image_id": "a", "native_score": 9, "themes": ["landscape"]},
        {"dataset_id": "tad66k", "image_id": "b", "native_score": 4, "themes": ["landscape"]},
        {"dataset_id": "tad66k", "image_id": "c", "native_score": 3, "themes": ["portrait"]},
        {"dataset_id": "other", "image_id": "d", "native_score": 4},
        {"dataset_id": "other", "image_id": "e", "native_score": 2},
        {"dataset_id": "other", "image_id": "f", "native_score": 9, "locked_test": True},
        {"dataset_id": "other", "image_id": "g", "native_score": 1},
    ]
    first = build_ranking_pairs(rows, seed=7)
    second = build_ranking_pairs(rows, seed=7)
    assert first == second
    assert [(pair["preferred_id"], pair["other_id"]) for pair in first] == [("d", "e"), ("d", "g"), ("e", "g"), ("a", "b")]
    assert all(pair["dataset_id"] != "tad66k" or pair["theme"] == "landscape" for pair in first)
    assert all("f" not in (pair["preferred_id"], pair["other_id"]) for pair in first)


def test_ranking_pair_budget_does_not_expand_quadratically():
    rows=[{"dataset_id":"aadb","image_id":str(i),"native_score":float(i)} for i in range(5000)]
    pairs=build_ranking_pairs(rows,seed=7,max_pairs=1000)
    assert len(pairs) <= 1000
    assert all(pair["dataset_id"] == "aadb" for pair in pairs)


def test_spaq_splits_keep_groups_together_and_are_deterministic():
    rows = []
    for group in range(10):
        for image in range(2):
            rows.append({
                "dataset_id": "spaq", "image_id": f"spaq:{group}:{image}",
                "native_score": 20 + group, "native_score_range": [0, 100],
                "group_id": f"phone-scene-{group}",
            })
    first = prepare_spaq_technical_splits(rows, folds=5, seed=17)
    second = prepare_spaq_technical_splits(rows, folds=5, seed=17)
    assert first == second
    assert len({row["development_fold"] for row in first if not row["locked_test"]}) == 5
    grouped = {}
    for row in first:
        grouped.setdefault(row["group_id"], set()).add((row["locked_test"], row["development_fold"]))
    assert all(len(values) == 1 for values in grouped.values())
    assert all(row["task"] == "technical" and row["eligible_as_aesthetic_target"] is False for row in first)
    assert all(row["technical_target"] == (row["native_score"] / 100) for row in first)
