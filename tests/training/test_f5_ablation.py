import torch

from image_aesthetics.models.fusion import build_baseline
from image_aesthetics.models.fusion.losses import pairwise_ranking_loss
from scripts.training.run_f5_ablation import (
    GATES,
    MODELS,
    PAIR_GAP,
    SEEDS,
    VERSION,
    FOLDS,
    RunSpec,
    _atomic_identity,
    _comparison,
    _make_model,
    _protocol,
    _parser,
    ablation_specs,
    plan_matrix,
)


def test_ablation_matrix_is_exact_12_runs_and_pairs_folds_seeds():
    specs = ablation_specs()
    assert len(specs) == 12
    assert tuple(sorted({spec.fold for spec in specs})) == FOLDS == (2, 3)
    assert tuple(sorted({spec.seed for spec in specs})) == SEEDS == (20260920, 20260921)
    assert {spec.model for spec in specs} == set(MODELS)
    assert specs[0].run_id == "F5-G-R15-fold2-seed20260920"
    assert specs[-1].run_id == "F5-G-Pair05-fold3-seed20260921"


def test_variants_have_expected_residual_and_gate_capacity():
    r15 = _make_model("F5-G-R15")
    no_gate = _make_model("F5-G-NoGate")
    pair = _make_model("F5-G-Pair05")
    assert r15.residual_bound == 0.15
    assert pair.residual_bound == 0.25
    assert no_gate.gate is None
    assert no_gate.gate_values(torch.zeros(3, 1544)).eq(1).all()
    assert no_gate.param_count == build_baseline("F5-G").param_count - 4392
    assert no_gate.architecture_config()["content_gate"] is False


def test_pair05_filters_only_normalized_label_gaps_below_threshold():
    scores = torch.tensor([0.8, 0.2])
    close_targets = torch.tensor([0.51, 0.49])
    assert pairwise_ranking_loss(scores, close_targets).item() > 0
    assert pairwise_ranking_loss(scores, close_targets, min_target_gap=PAIR_GAP).item() == 0
    separated_targets = torch.tensor([0.53, 0.47])
    assert pairwise_ranking_loss(scores, separated_targets,
                                 min_target_gap=PAIR_GAP).item() > 0


def test_run_identity_binds_control_architecture_and_pairing_protocol():
    spec = RunSpec("F5-G-Pair05-fold2-seed20260920", "F5-G-Pair05", 2, 20260920)
    common = dict(split_sha256="split", normalization_sha256="norm",
                  cache_identity={"aadb": "cache"},
                  architecture={"model_id": "F5-G-v1", "residual_bound": 0.25},
                  initialization={"run_id": "F2-fold2-seed20260920"},
                  protocol={"ranking_min_normalized_target_gap": 0.05})
    ref = {"run_id": "F5-G-fold2-seed20260920", "identity": "g-id",
           "checkpoint_sha256": "checkpoint", "config_sha256": "config"}
    identity, payload = _atomic_identity(spec, control_reference=ref, **common)
    changed, _ = _atomic_identity(spec, control_reference={**ref, "identity": "other"}, **common)
    assert identity != changed
    assert payload["experiment_version"] == VERSION
    assert payload["locked_test_accessed"] is False
    assert payload["matched_f5_g_control"] == ref


def test_protocol_only_pair05_uses_point_zero_five_normalized_gap():
    args = type("Args", (), {"device": "mps", "batch_size": 1024,
                             "num_workers": 4, "status_interval_batches": 25,
                             "epochs": 50, "patience": 7, "min_epochs": 8})()
    pair = _protocol(args, ablation_specs()[-1])
    other = _protocol(args, ablation_specs()[0])
    assert pair["ranking_min_normalized_target_gap"] == 0.05
    assert other["ranking_min_normalized_target_gap"] == 0.0
    assert pair["technical_auxiliary"] is False
    assert pair["signal_dropout"] is False


def test_plan_is_read_only_and_records_predeclared_gates(tmp_path, monkeypatch):
    import scripts.training.run_f5_ablation as runner

    monkeypatch.setattr(runner, "_load_g_controls", lambda path: ({
        f"F5-G-fold{fold}-seed{seed}": {"identity": "id"}
        for fold in FOLDS for seed in SEEDS
    }, "digest"))
    args = type("Args", (), {
        "data_root": tmp_path / "data",
        "baseline_root": tmp_path / "baseline",
        "control_root": tmp_path / "controls",
        "output_root": tmp_path / "external" / VERSION,
        "device": "mps", "batch_size": 1024, "num_workers": 4,
        "status_interval_batches": 25, "epochs": 50, "patience": 7, "min_epochs": 8,
    })()
    plan = plan_matrix(args)
    assert plan["total"] == 12
    assert plan["locked_test_accessed"] is False
    assert plan["controls_reused"] == 4
    assert plan["predeclared_gates"] == GATES
    assert plan["ranking_min_normalized_target_gap_by_model"] == {
        "F5-G-R15": 0.0, "F5-G-NoGate": 0.0, "F5-G-Pair05": 0.05,
    }
    assert [run["protocol"]["ranking_min_normalized_target_gap"]
            for run in plan["runs"][:3]] == [0.0, 0.0, 0.05]
    assert plan["runs"][0]["model_config"]["residual_bound"] == 0.15
    assert not args.output_root.exists()


def test_comparison_aggregates_per_dataset_metrics():
    import scripts.training.run_f5_ablation as runner

    old_datasets, old_metrics = runner.DATASETS, runner.METRICS
    runner.DATASETS, runner.METRICS = ("setA",), ("spearman",)
    try:
        candidates = {
            f"{model}-fold2-seed20260920": {
                "model": model, "fold": 2, "seed": 20260920,
                "best_metrics": {"val": {
                    "macro": {"spearman": 0.6},
                    "per_dataset": {"setA": {"spearman": 0.7}},
                }},
            }
            for model in MODELS
        }
        references = {
            f"{model}-fold2-seed20260920": {
                "model": model, "fold": 2, "seed": 20260920,
                "best_metrics": {"val": {
                    "macro": {"spearman": 0.5},
                    "per_dataset": {"setA": {"spearman": 0.4}},
                }},
            }
            for model in ("F2", "F4")
        }
        report = _comparison(candidates, candidates, references)
        assert report["by_model"]["F5-G-R15"]["per_dataset"]["setA"]["spearman"] == {
            "n": 1, "mean": 0.7, "sd": 0.0,
        }
        delta = report["paired_vs"]["F5-G-R15"]["vs_F4"]["pairs"][0]
        assert abs(delta["per_dataset_delta"]["setA"]["spearman"] - 0.3) < 1e-12
    finally:
        runner.DATASETS, runner.METRICS = old_datasets, old_metrics


def test_runner_exposes_no_locked_test_option():
    help_text = _parser().format_help()
    assert "--locked-test" not in help_text
    assert "--allow-locked-test" not in help_text
