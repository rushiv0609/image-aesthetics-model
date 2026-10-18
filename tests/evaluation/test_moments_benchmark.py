import pytest

from image_aesthetics.evaluation.moments_benchmark import evaluate_moments_predictions


def _manifest():
    return [{"item_id": f"i{i}", "event_id": "e1" if i < 3 else "e2"}
            for i in range(6)]


def test_metrics_use_real_annotations_and_event_local_selection():
    annotations = [
        {"kind": "pairwise", "left_id": "i0", "right_id": "i1", "winner_id": "i0"},
        {"kind": "keep_reject", "item_id": "i0", "keep": True},
        {"kind": "keep_reject", "item_id": "i3", "keep": False},
        {"kind": "topk", "event_id": "e1", "task_id": "t1", "selected_ids": ["i0"]},
    ]
    predictions = [{"model": "F5-G", "item_id": f"i{i}", "score": score}
                   for i, score in enumerate([.9, .2, .1, .8, .3, .2])]
    report = evaluate_moments_predictions(_manifest(), annotations, predictions)
    metrics = report["models"]["F5-G"]
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["top_fractions"]["0.05"]["selected_n"] == 2
    assert metrics["top_fractions"]["0.05"]["severe_false_positive_rate"] == .5
    assert report["human_labels_present"] is True


def test_missing_human_labels_are_pending_not_fabricated():
    predictions = [{"model": "F4", "item_id": f"i{i}", "score": .5}
                   for i in range(6)]
    report = evaluate_moments_predictions(_manifest(), [], predictions)
    assert report["human_labels_present"] is False
    assert report["models"]["F4"]["pairwise_accuracy"] is None
    assert report["models"]["F4"]["top_fractions"]["0.01"]["severe_false_positive_rate"] is None


def test_cross_event_pair_is_rejected():
    annotations = [{"kind": "pairwise", "left_id": "i0", "right_id": "i3",
                    "winner_id": "i0"}]
    predictions = [{"model": "F5-G", "item_id": f"i{i}", "score": .5}
                   for i in range(6)]
    with pytest.raises(ValueError, match="crosses events"):
        evaluate_moments_predictions(_manifest(), annotations, predictions)
