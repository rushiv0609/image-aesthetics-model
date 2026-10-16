import ast
import json
from pathlib import Path

import pytest

from scripts.training.run_f5_screen import (
    FOLDS, MODELS, SEEDS, _matrix_status, _parser, artifact_paths,
    _assert_finite, _validate_roots, paired_analysis, plan_matrix,
    reusable_completed_run, run_identity,
    screen_specs,
)


def test_screen_has_exact_fixed_16_runs():
    specs = screen_specs()
    assert len(specs) == 16
    assert tuple(sorted({spec.fold for spec in specs})) == FOLDS
    assert tuple(sorted({spec.seed for spec in specs})) == SEEDS
    assert {spec.model for spec in specs} == set(MODELS)
    assert [spec.run_id for spec in specs[:4]] == [
        "F3C-fold0-seed20260920", "F5-R-fold0-seed20260920",
        "F5-G-fold0-seed20260920", "F5-X-fold0-seed20260920",
    ]
    assert all(spec.run_id.startswith(f"{spec.model}-fold{spec.fold}-seed{spec.seed}")
               for spec in specs)


def test_identity_is_stable_and_binds_variant_source_and_protocol():
    spec = screen_specs()[1]
    kwargs = dict(
        split_sha256="split", normalization_sha256="norm",
        cache_identity={"aadb": "joined-a", "para": "joined-p"},
        architecture={"model_id": "F5-R-v1"},
        f2_initialization={"identity": "fold0-seed20260920-source"},
        protocol={"device": "mps", "batch_size": 1024},
    )
    first, payload = run_identity(spec, **kwargs)
    second, _ = run_identity(spec, **kwargs)
    assert first == second
    assert payload["locked_test_accessed"] is False
    assert payload["technical_auxiliary"] is False
    assert payload["signal_dropout"] is False
    for key, changed in [
        ("normalization_sha256", "other-norm"),
        ("split_sha256", "other-split"),
    ]:
        modified = dict(kwargs)
        modified[key] = changed
        assert run_identity(spec, **modified)[0] != first
    modified = dict(kwargs)
    modified["f2_initialization"] = {"identity": "other-fold-seed-source"}
    assert run_identity(spec, **modified)[0] != first
    modified = dict(kwargs)
    modified["protocol"] = {"device": "mps", "batch_size": 512}
    assert run_identity(spec, **modified)[0] != first


def test_artifact_paths_reuse_baseline_fold_normalizer_and_f2(tmp_path):
    spec = next(s for s in screen_specs()
                if s.model == "F5-X" and s.fold == 4 and s.seed == 20260921)
    paths = artifact_paths(tmp_path, tmp_path / "baseline-matrix-v6", spec)
    assert paths["split_manifest"] == tmp_path / "fusion/prepared/v6/samples.jsonl"
    assert paths["normalization"] == tmp_path / "baseline-matrix-v6/normalization/fold4.json"
    assert paths["f2_checkpoint"] == (
        tmp_path / "baseline-matrix-v6/F2-fold4-seed20260921/checkpoints/best.pt")
    assert set(paths["joins"]) == {"aadb", "para", "flickr_aes", "tad66k"}
    assert all(str(path).endswith("features/joined-ten-v1")
               for path in paths["joins"].values())


def test_dataset_construction_has_only_train_and_validation_partitions():
    source = Path("scripts/training/run_f5_screen.py").read_text()
    tree = ast.parse(source)
    partitions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "FusionTrainingDataset":
            for keyword in node.keywords:
                if keyword.arg == "partition" and isinstance(keyword.value, ast.Constant):
                    partitions.add(keyword.value.value)
    assert partitions == {"train", "validation"}


def test_plan_is_exact_and_read_only(tmp_path):
    args = _parser().parse_args(["--data-root", str(tmp_path),
                                 "--baseline-root", str(tmp_path / "baseline")])
    args.output_root = tmp_path / "external" / "f5-screen-v1"
    plan = plan_matrix(args)
    assert plan["total"] == 16
    assert plan["locked_test_accessed"] is False
    assert plan["technical_auxiliary"] is False
    assert plan["signal_dropout"] is False
    assert not args.output_root.exists()
    assert args.device == "mps" and args.batch_size == 1024 and args.num_workers == 4


def test_output_root_cannot_overlap_repo_or_baseline(tmp_path):
    args = _parser().parse_args([])
    args.baseline_root = tmp_path / "baseline"
    args.data_root = tmp_path / "data"
    with pytest.raises(ValueError, match="baseline root"):
        _validate_roots(args, args.baseline_root / "f5-screen-v1")


def test_completed_run_reuse_requires_matching_all_artifact_identities(tmp_path):
    identity = "run-identity"
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"identity": identity}))
    (run / "training-status.json").write_text(json.dumps({
        "state": "completed", "identity": identity,
    }))
    (checkpoints / "checkpoint.json").write_text(json.dumps({"identity": identity}))
    (checkpoints / "best.pt").write_bytes(b"best")
    (checkpoints / "last.pt").write_bytes(b"last")
    assert reusable_completed_run(run, identity)
    assert not reusable_completed_run(run, "different-identity")
    (checkpoints / "checkpoint.json").write_text(json.dumps({"identity": "wrong"}))
    assert not reusable_completed_run(run, identity)


def test_matrix_status_is_atomic_and_tracks_pending_runs(tmp_path):
    specs = screen_specs()
    status = _matrix_status(tmp_path / "matrix-status.json", state="running",
                            specs=specs, completed=[specs[0].run_id],
                            failed=[], current=specs[1].run_id, started_at=1.0)
    assert status["total"] == 16
    assert status["completed"] == [specs[0].run_id]
    assert specs[0].run_id not in status["pending"]
    assert specs[1].run_id in status["pending"]
    assert status["locked_test_accessed"] is False
    assert (tmp_path / "matrix-status.json").is_file()
    assert not (tmp_path / "matrix-status.json.tmp").exists()


def test_analysis_pairs_each_candidate_to_same_fold_and_seed_references():
    def report(value):
        macro = {name: value for name in (
            "spearman", "pairwise_accuracy", "top1_lift", "top5_lift",
            "top10_lift", "top1_overlap", "top5_overlap", "top10_overlap",
            "mae", "rmse")}
        return {"macro": macro,
                "per_dataset": {"aadb": macro, "para": macro,
                                "flickr_aes": macro, "tad66k": macro}}

    candidate = {"run_id": "F5-R-fold0-seed20260920", "model": "F5-R",
                 "fold": 0, "seed": 20260920,
                 "best_metrics": report(0.8)}
    baseline = {
        "F2-fold0-seed20260920": {"val": report(0.6)},
        "F4-fold0-seed20260920": {"val": report(0.7)},
    }
    result = paired_analysis({candidate["run_id"]: candidate}, baseline)
    versus_f2 = result["paired_vs_baseline"]["F5-R"]["vs_F2"]
    versus_f4 = result["paired_vs_baseline"]["F5-R"]["vs_F4"]
    assert versus_f2["n_pairs"] == versus_f4["n_pairs"] == 1
    assert versus_f2["pairs"][0]["reference_run_id"] == "F2-fold0-seed20260920"
    assert versus_f4["pairs"][0]["reference_run_id"] == "F4-fold0-seed20260920"
    assert versus_f2["metrics"]["spearman"]["mean"] == pytest.approx(0.2)
    assert versus_f4["metrics"]["spearman"]["mean"] == pytest.approx(0.1)
    assert versus_f2["metrics"]["spearman"]["improved_pairs"] == 1
    assert versus_f2["metrics"]["mae"]["improved_pairs"] == 0


def test_nonfinite_metrics_are_rejected():
    with pytest.raises(ValueError, match="non-finite metric"):
        _assert_finite({"macro": {"spearman": float("nan")}})
