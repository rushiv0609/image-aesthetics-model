from image_aesthetics.data.fusion_preparation import collapse_feature_labels


def test_collapse_feature_labels_uses_one_aggregate_target_per_image():
    joined = [{"image_id": "flickr:x", "label_join_id": "flickr:x", "dataset_id": "flickr",
               "source_sha256": "a" * 64, "eligible_for_fusion": True,
               "eligible_as_aesthetic_target": True, "quarantine": False,
               "locked_test": True}]
    labels = [{"label_join_id": "flickr:x", "sha256": "a" * 64, "image_name": "x.jpg",
               "metadata": {"aggregate_score_0_1": 0.7}, "source_groups": ["image:x"]},
              {"label_join_id": "flickr:x", "sha256": "a" * 64, "image_name": "x.jpg",
               "metadata": {"aggregate_score_0_1": 0.7}, "source_groups": ["image:x"]}]
    rows, audit = collapse_feature_labels(labels, joined)
    assert len(rows) == 1
    assert rows[0]["native_score"] == 0.7
    assert rows[0]["native_score_range"] == [0.0, 1.0]
    assert rows[0]["raw_label_count"] == 2
    assert rows[0]["upstream_locked_test"] is True
    assert audit["eligible_targets"] == 1


def test_collapse_feature_labels_excludes_non_aesthetic_and_quarantined_rows():
    labels = [{"label_join_id": "x", "score": 0.5, "path": "x.jpg"}]
    base = {"image_id": "x", "label_join_id": "x", "dataset_id": "fixture",
            "source_sha256": "a" * 64, "eligible_for_fusion": True,
            "eligible_as_aesthetic_target": True, "quarantine": False}
    variants = [{**base, "eligible_as_aesthetic_target": False}, {**base, "quarantine": True}]
    rows, audit = collapse_feature_labels(labels, variants)
    assert rows == []
    assert audit["eligible_targets"] == 0


def test_collapse_feature_labels_averages_raw_rater_scores():
    joined = [{"image_id": "flickr:x", "label_join_id": "flickr:x", "dataset_id": "flickr",
               "source_sha256": "a" * 64, "eligible_for_fusion": True,
               "eligible_as_aesthetic_target": True, "quarantine": False}]
    labels = [{"label_join_id": "flickr:x", "sha256": "a" * 64, "image_name": "x.jpg",
               "native_score": score, "native_score_range": [1.0, 5.0]}
              for score in (2.0, 3.0, 4.0)]
    rows, audit = collapse_feature_labels(labels, joined)
    assert rows[0]["native_score"] == 3.0
    assert rows[0]["raw_label_count"] == 3
    assert audit["inconsistent_label_count"] == 0
