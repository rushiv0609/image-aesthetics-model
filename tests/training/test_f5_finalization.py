import json
import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest

from image_aesthetics.training.f5_finalization import (
    claim_locked_evaluation, freeze_policy, load_frozen_policy, validate_policy,
)


HASH = "a" * 64


def _locked_evaluation_module():
    path = Path(__file__).resolve().parents[2] / "scripts/evaluation/evaluate_f5_locked_once.py"
    spec = importlib.util.spec_from_file_location("evaluate_f5_locked_once_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def policy():
    return {
        "schema_version": "f5-frozen-policy-v1",
        "candidate": "F5-G",
        "epochs": {"f2": 9, "candidate": 12},
        "seeds": [20260920, 20260921],
        "calibration": {"choice": "shared-isotonic-v1", "path": "/external/cal.json",
                        "sha256": HASH, "oof_predictions_sha256": "f" * 64},
        "feature_hashes": {"aadb": HASH, "para": "b" * 64},
        "split_sha256": "c" * 64,
        "thresholds": {"minimum_macro_spearman_delta": 0.0},
        "evidence_hashes": {"screen": "d" * 64, "lodo": "e" * 64},
        "architecture": {"model_id": "F5-G-v1", "kwargs": {}},
        "preprocessing": {"feature_contract": "joined-ten-v1"},
        "objective": {"aesthetic_loss": "calibrated-huber",
                       "ranking_loss": "compatible-pairwise"},
        "checkpoint_policy": {"mode": "fixed-epoch-final", "early_stopping": False},
        "training": {"batch_size": 256, "datasets": ["aadb", "para"],
                     "ranking_weight": 0.1, "ranking_min_target_gap": 0.0,
                     "device": "mps"},
    }


def test_frozen_policy_is_written_once_and_sidecar_is_verified(tmp_path):
    path = tmp_path / "frozen-policy.json"
    digest = freeze_policy(path, policy())
    loaded, verified = load_frozen_policy(path)
    assert digest == verified
    assert loaded["candidate"] == "F5-G"
    with pytest.raises(FileExistsError):
        freeze_policy(path, policy())


def test_policy_requires_calibration_thresholds_and_evidence_hashes():
    value = policy()
    del value["thresholds"]
    with pytest.raises(ValueError, match="thresholds"):
        validate_policy(value)


def test_policy_digest_sidecar_detects_mutation(tmp_path):
    path = tmp_path / "policy.json"
    freeze_policy(path, policy())
    path.chmod(0o644)
    changed = policy()
    changed["thresholds"]["minimum_macro_spearman_delta"] = 0.01
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest"):
        load_frozen_policy(path)


def test_locked_test_receipt_is_exclusive_and_consumed_on_failure(tmp_path):
    receipt = tmp_path / "one-time-receipt.json"
    value = claim_locked_evaluation(receipt, policy_sha256=HASH,
                                    fit_manifest_sha256="b" * 64)
    assert value["state"] == "started"
    assert json.loads(receipt.read_text())["locked_test_access_started"] is True
    with pytest.raises(FileExistsError):
        claim_locked_evaluation(receipt, policy_sha256=HASH,
                                fit_manifest_sha256="b" * 64)


def test_non_locked_preflight_failure_does_not_consume_receipt(tmp_path, monkeypatch):
    locked_evaluation = _locked_evaluation_module()
    data_root = tmp_path / "data"
    fit_root = data_root / "fusion/experiments/final-fit"
    report = data_root / "fusion/experiments/final-report.json"
    policy_path = tmp_path / "policy.json"
    data_root.mkdir(); fit_root.mkdir(parents=True)
    freeze_policy(policy_path, policy())
    monkeypatch.setattr(
        locked_evaluation, "_verify_fit",
        lambda *_args, **_kwargs: ({"normalization_sha256": HASH}, HASH),
    )
    monkeypatch.setattr(
        locked_evaluation, "_verify_non_locked_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("safe preflight rejected corrupt calibration")),
    )
    monkeypatch.setattr(
        locked_evaluation, "claim_locked_evaluation",
        lambda *_args, **_kwargs: pytest.fail("receipt claimed before safe preflight"),
    )
    args = SimpleNamespace(
        policy=policy_path, data_root=data_root, fit_root=fit_root,
        receipt=None, report=report, device=None, batch_size=8, num_workers=0,
    )
    with pytest.raises(ValueError, match="safe preflight"):
        locked_evaluation.execute(args)
    assert not locked_evaluation._canonical_receipt(data_root).exists()
