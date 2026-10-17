import ast
import json
from pathlib import Path

import pytest

from scripts.training.run_f5_confirmation import (
    FOLDS,
    MODELS,
    SEEDS,
    VERSION,
    _identity,
    _parser,
    _prior_failure_blocks_retry,
    _validate_confirmation_roots,
    _write_matrix_status,
    artifact_paths,
    confirmation_specs,
    plan_matrix,
    reusable_completed_run,
)


def test_confirmation_matrix_is_exact_30_runs():
    specs = confirmation_specs()
    assert len(specs) == 30
    assert tuple(sorted({spec.fold for spec in specs})) == FOLDS
    assert tuple(sorted({spec.seed for spec in specs})) == SEEDS
    assert {spec.model for spec in specs} == set(MODELS) == {"F5-G", "F5-X"}
    assert specs[0].run_id == "F5-G-fold0-seed20260920"
    assert specs[1].run_id == "F5-X-fold0-seed20260920"
    assert specs[-1].run_id == "F5-X-fold4-seed20260922"


def test_paths_bind_each_run_to_matching_fold_and_seed_f2(tmp_path):
    root = tmp_path / "baseline-matrix-v6"
    paths_by_seed = []
    for spec in confirmation_specs():
        if spec.fold == 2 and spec.model == "F5-X":
            paths_by_seed.append(artifact_paths(tmp_path / "data", root, spec)["f2_checkpoint"])
    assert paths_by_seed == [
        root / "F2-fold2-seed20260920/checkpoints/best.pt",
        root / "F2-fold2-seed20260921/checkpoints/best.pt",
        root / "F2-fold2-seed20260922/checkpoints/best.pt",
    ]
    assert len(set(paths_by_seed)) == 3


def test_identity_is_versioned_and_binds_seed_initialization():
    spec = confirmation_specs()[0]
    common = dict(
        split_sha256="split", normalization_sha256="fold-norm",
        cache_identity={"aadb": "cache-a"}, architecture={"model_id": "F5-G"},
        protocol={"device": "mps", "batch_size": 1024},
    )
    identity_a, payload = _identity(
        spec, initialization={"source": "F2-fold0-seed20260920"}, **common)
    identity_b, _ = _identity(
        spec, initialization={"source": "F2-fold0-seed20260921"}, **common)
    assert identity_a != identity_b
    assert payload["experiment_version"] == VERSION
    assert payload["locked_test_accessed"] is False


def test_resume_reuses_only_matching_completed_artifacts_and_failed_requires_flag(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints"
    checkpoint.mkdir(parents=True)
    identity = "stable"
    (run / "config.json").write_text(json.dumps({"identity": identity}))
    (run / "training-status.json").write_text(json.dumps({
        "state": "completed", "identity": identity,
    }))
    (checkpoint / "checkpoint.json").write_text(json.dumps({"identity": identity}))
    (checkpoint / "best.pt").write_bytes(b"best")
    (checkpoint / "last.pt").write_bytes(b"last")
    assert reusable_completed_run(run, identity)
    assert not reusable_completed_run(run, "changed")
    failed = {"state": "failed"}
    assert _prior_failure_blocks_retry(failed, retry_failed=False)
    assert not _prior_failure_blocks_retry(failed, retry_failed=True)
    assert not _prior_failure_blocks_retry({"state": "running"}, retry_failed=False)


def test_plan_is_read_only_and_uses_separate_root_and_fixed_aesthetic_protocol(tmp_path):
    args = _parser().parse_args([
        "--data-root", str(tmp_path / "data"),
        "--baseline-root", str(tmp_path / "baseline"),
    ])
    args.output_root = tmp_path / "external" / "f5-confirmation-v1"
    plan = plan_matrix(args)
    assert plan["total"] == 30
    assert plan["experiment_version"] == VERSION
    assert plan["output_root"].endswith("f5-confirmation-v1")
    assert plan["locked_test_accessed"] is False
    assert plan["technical_auxiliary"] is False
    assert plan["signal_dropout"] is False
    assert plan["protocol"]["device"] == "mps"
    assert plan["protocol"]["batch_size"] == 1024
    assert plan["protocol"]["num_workers"] == 4
    assert not args.output_root.exists()


def test_confirmation_root_cannot_overlap_screen_artifacts(tmp_path):
    args = _parser().parse_args([
        "--data-root", str(tmp_path / "data"),
        "--baseline-root", str(tmp_path / "baseline"),
    ])
    screen_root = args.data_root / "fusion/experiments/f5-screen-v1"
    with pytest.raises(ValueError, match="screen artifacts"):
        _validate_confirmation_roots(args, screen_root)


def test_matrix_status_reports_exact_pending_and_never_claims_test_access(tmp_path):
    specs = confirmation_specs()
    status = _write_matrix_status(
        tmp_path / "matrix-status.json", state="running", specs=specs,
        completed=[specs[0].run_id], failed=[], current=specs[1].run_id,
        started_at=1.0)
    assert status["experiment_version"] == VERSION
    assert status["total"] == 30
    assert status["pending"][0] == specs[1].run_id
    assert status["locked_test_accessed"] is False
    persisted = json.loads((tmp_path / "matrix-status.json").read_text())
    assert persisted == status
    assert not (tmp_path / "matrix-status.json.tmp").exists()


def test_runner_constructs_only_train_and_validation_partitions():
    source = Path("scripts/training/run_f5_confirmation.py").read_text()
    tree = ast.parse(source)
    partition_values = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "FusionTrainingDataset":
            for keyword in node.keywords:
                if keyword.arg == "partition" and isinstance(keyword.value, ast.Constant):
                    partition_values.add(keyword.value.value)
    # Dataset creation is delegated to the established screen helper whose
    # source explicitly constructs train and validation partitions. This runner
    # has no test-partition selector or override.
    assert partition_values == set()
    assert "--locked-test" not in source
    screen_source = Path("scripts/training/run_f5_screen.py").read_text()
    screen_tree = ast.parse(screen_source)
    screen_partitions = set()
    for node in ast.walk(screen_tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "FusionTrainingDataset":
            for keyword in node.keywords:
                if keyword.arg == "partition" and isinstance(keyword.value, ast.Constant):
                    screen_partitions.add(keyword.value.value)
    assert screen_partitions == {"train", "validation"}
