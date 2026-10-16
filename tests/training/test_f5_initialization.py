import hashlib
import json

import pytest
import torch

from image_aesthetics.models.fusion import F5R, RegularizedLinear
from image_aesthetics.training.f5_initialization import initialize_from_fold_f2


def _source(tmp_path, *, fold=2, seed=20260921):
    run = tmp_path / f"F2-fold{fold}-seed{seed}"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    split_path = tmp_path / "samples.jsonl"
    split_path.write_text('{"split":"frozen"}\n')
    normalization = tmp_path / f"fold{fold}.json"
    normalization.write_text('{"normalizer":"fold-specific"}\n')
    identity = "source-identity"
    payload = {
        "spec": {"model": "F2", "fold": fold, "seed": seed},
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "normalization_stats_sha256": hashlib.sha256(normalization.read_bytes()).hexdigest(),
        "normalize": True,
        "limit": None,
    }
    (run / "config.json").write_text(json.dumps({
        "model": "F2", "run_id": f"F2-fold{fold}-seed{seed}",
        "identity": identity, "identity_payload": payload,
        "locked_test_accessed": False,
    }))
    (checkpoints / "checkpoint.json").write_text(json.dumps({
        "identity": identity, "best_epoch": 7,
    }))
    torch.manual_seed(333)
    f2 = RegularizedLinear()
    checkpoint = checkpoints / "best.pt"
    torch.save({"model": f2.state_dict()}, checkpoint)
    return checkpoint, split_path, normalization, f2


def test_f2_checkpoint_provenance_and_scalar_copy(tmp_path):
    checkpoint, split, normalization, f2 = _source(tmp_path)
    model = F5R()
    result = initialize_from_fold_f2(
        model, checkpoint, fold=2, seed=20260921,
        split_manifest=split, normalization_path=normalization)
    assert result["model_id"] == "F2"
    assert result["fold"] == 2 and result["seed"] == 20260921
    assert result["locked_test_accessed"] is False
    assert model.f2_initialization_loaded
    assert torch.equal(model.scalar_base.weight, f2.linear.weight)
    assert torch.equal(model.scalar_base.bias, f2.linear.bias)


def test_f2_checkpoint_rejects_wrong_fold_seed_or_normalizer(tmp_path):
    checkpoint, split, normalization, _ = _source(tmp_path)
    for kwargs, message in [
        ({"fold": 1, "seed": 20260921}, "fold/seed"),
        ({"fold": 2, "seed": 20260920}, "fold/seed"),
    ]:
        with pytest.raises(ValueError, match=message):
            initialize_from_fold_f2(
                F5R(), checkpoint, split_manifest=split,
                normalization_path=normalization, **kwargs)
    changed = tmp_path / "changed.json"
    changed.write_text("different normalization")
    with pytest.raises(ValueError, match="normalization"):
        initialize_from_fold_f2(
            F5R(), checkpoint, fold=2, seed=20260921,
            split_manifest=split, normalization_path=changed)


def test_f2_checkpoint_rejects_identity_mismatch_and_locked_test(tmp_path):
    checkpoint, split, normalization, _ = _source(tmp_path)
    checkpoint_meta = checkpoint.parent / "checkpoint.json"
    checkpoint_meta.write_text(json.dumps({"identity": "wrong"}))
    with pytest.raises(ValueError, match="identity"):
        initialize_from_fold_f2(
            F5R(), checkpoint, fold=2, seed=20260921,
            split_manifest=split, normalization_path=normalization)
    checkpoint_meta.write_text(json.dumps({"identity": "source-identity"}))
    config_path = checkpoint.parent.parent / "config.json"
    config = json.loads(config_path.read_text())
    config["locked_test_accessed"] = True
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="locked-test"):
        initialize_from_fold_f2(
            F5R(), checkpoint, fold=2, seed=20260921,
            split_manifest=split, normalization_path=normalization)
