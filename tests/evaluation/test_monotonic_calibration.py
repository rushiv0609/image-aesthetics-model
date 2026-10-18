import pytest
import json

from image_aesthetics.evaluation.monotonic_calibration import fit_isotonic, validate_oof_rows
from scripts.evaluation.evaluate_shared_calibration import _metrics, evaluate
from scripts.evaluation.export_f5_oof_predictions import _validated_run


def _rows():
    rows = []
    for fold in range(3):
        for index in range(3):
            rows.append({"dataset_id": "aadb", "image_id": f"{fold}-{index}",
                         "group_id": f"g-{fold}-{index}", "development_fold": fold,
                         "source_validation_fold": fold, "prediction": index / 3,
                         "target": (index + fold * .1) / 3.2, "locked_test": False})
    return rows


def test_pav_mapping_is_monotonic_and_bounded():
    calibrator = fit_isotonic([0.1, 0.2, 0.3, 0.4], [0.2, 0.8, 0.4, 0.9])
    values = calibrator.predict([0.1, 0.2, 0.3, 0.4])
    assert list(values) == sorted(values)
    assert min(values) >= 0 and max(values) <= 1


def test_pav_tied_scores_share_their_mean_fit_value():
    calibrator = fit_isotonic([0.5, 0.5, 0.8], [0.1, 0.9, 0.2])
    assert calibrator.predict([0.5])[0] == pytest.approx(0.4)
    assert calibrator.thresholds == (0.8,)


def test_oof_validator_rejects_locked_and_in_fold_mismatch():
    rows = _rows()
    rows[0]["locked_test"] = True
    with pytest.raises(ValueError, match="locked-test"):
        validate_oof_rows(rows)
    rows = _rows(); rows[0]["source_validation_fold"] = 2
    with pytest.raises(ValueError, match="not out-of-fold"):
        validate_oof_rows(rows)


def test_cross_fit_uses_other_folds_and_preserves_rank_metric():
    report, payload, outputs = evaluate(_rows())
    assert report["locked_test_accessed"] is False
    assert report["dataset_identity_used_as_input"] is False
    assert len(outputs) == 9
    assert payload["schema_version"] == "shared-isotonic-v1"
    assert all(row["calibrated_prediction"] >= 0 for row in outputs)


def test_calibration_spearman_uses_average_ranks_for_ties():
    report = _metrics([0.5, 0.5, 0.9], [0.1, 0.5, 0.9])
    assert report["spearman"] == pytest.approx(0.8660254037844387)


def test_oof_export_rejects_wrong_candidate_or_missing_lock_boundary(tmp_path):
    run_id = "F5-G-fold0-seed20260920"
    run = tmp_path / run_id
    checkpoint = run / "checkpoints"
    checkpoint.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({
        "run_id": run_id, "model": "F5-X", "fold": 0, "seed": 20260920,
        "identity": "same", "locked_test_accessed": False,
    }))
    (run / "training-status.json").write_text(json.dumps({
        "state": "completed", "identity": "same", "locked_test_accessed": False,
    }))
    (checkpoint / "checkpoint.json").write_text(json.dumps({"identity": "same"}))
    (checkpoint / "best.pt").write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        _validated_run(tmp_path, "F5-G", 0, 20260920)
    config = json.loads((run / "config.json").read_text())
    config["model"] = "F5-G"
    (run / "config.json").write_text(json.dumps(config))
    status = json.loads((run / "training-status.json").read_text())
    status["locked_test_accessed"] = True
    (run / "training-status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        _validated_run(tmp_path, "F5-G", 0, 20260920)
