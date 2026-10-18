#!/usr/bin/env python3
"""Plan or execute the fixed 12-run F5-G diagnostic ablation screen.

The matrix compares a smaller residual bound, removing the content gate, and
filtering weak ranking pairs. It uses folds 2/3 and seeds 20260920/20260921,
matching fold/seed F5-G confirmation controls. Only train/validation data is
constructed; there is no locked-test option. Artifacts live in the separate,
versioned f5-ablation-screen-v1 root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import DatasetCalibrationHeads, build_baseline
from image_aesthetics.training.f5_initialization import initialize_from_fold_f2, sha256_file
from image_aesthetics.training.fusion_trainer import train_fusion
from scripts.training.run_f5_confirmation import (
    _candidate_diagnostics,
    _load_baseline_reference,
    _mean_std,
    _prepare_inputs,
    _ranges,
    _read_json,
    _assert_finite,
    artifact_paths,
)
from scripts.training.run_f5_screen import (
    DATASETS,
    _atomic_json,
    _cache_hashes,
    _metric_values,
    _validate_roots as _validate_screen_roots,
    reusable_completed_run,
)

FOLDS = (2, 3)
SEEDS = (20260920, 20260921)
MODELS = ("F5-G-R15", "F5-G-NoGate", "F5-G-Pair05")
VERSION = "f5-ablation-screen-v1"
PAIR_GAP = 0.05
METRICS = ("spearman", "pairwise_accuracy", "top1_lift", "top5_lift",
           "top10_lift", "top1_overlap", "top5_overlap", "top10_overlap",
           "mae", "rmse")
GATES = {
    "macro_spearman_vs_F4_min": 0.030,
    "tad66k_spearman_delta_vs_F4_min_exclusive": -0.0125,
    "top5_lift_delta_vs_G_min": -0.002,
    "macro_mae_delta_vs_G_max": 0.002,
    "each_diagnostic_fold_spearman_delta_vs_G_min": 0.0,
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    fold: int
    seed: int


def ablation_specs() -> list[RunSpec]:
    return [RunSpec(f"{model}-fold{fold}-seed{seed}", model, fold, seed)
            for fold in FOLDS for seed in SEEDS for model in MODELS]


def _make_model(model_id: str) -> torch.nn.Module:
    if model_id == "F5-G-R15":
        return build_baseline("F5-G", residual_bound=0.15)
    if model_id == "F5-G-NoGate":
        return build_baseline("F5-G-NoGate", residual_bound=0.25)
    if model_id == "F5-G-Pair05":
        return build_baseline("F5-G", residual_bound=0.25)
    raise ValueError(f"unknown F5 ablation candidate: {model_id}")


def _atomic_identity(spec: RunSpec, *, split_sha256: str,
                     normalization_sha256: str,
                     cache_identity: dict[str, str],
                     architecture: dict[str, Any],
                     initialization: dict[str, Any],
                     control_reference: dict[str, str],
                     protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {
        "experiment_version": VERSION,
        "spec": asdict(spec),
        "split_manifest_sha256": split_sha256,
        "normalization_sha256": normalization_sha256,
        "joined_cache_identity": dict(sorted(cache_identity.items())),
        "architecture": architecture,
        "f2_initialization": initialization,
        "matched_f5_g_control": control_reference,
        "protocol": protocol,
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _protocol(args: argparse.Namespace, spec: RunSpec) -> dict[str, Any]:
    threshold = PAIR_GAP if spec.model == "F5-G-Pair05" else 0.0
    return {
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "status_interval_batches": args.status_interval_batches,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR after 5% warmup",
        "loss": "Huber(delta=1) + 0.1 within-compatible-group ranking",
        "ranking_min_normalized_target_gap": threshold,
        "technical_auxiliary": False,
        "signal_dropout": False,
        "limit": None,
    }


def _load_g_controls(path: Path, *, folds: tuple[int, ...] = FOLDS,
                     seeds: tuple[int, ...] = SEEDS) -> tuple[dict[str, dict[str, Any]], str]:
    aggregate = _read_json(path)
    if (aggregate.get("experiment_version") != "f5-confirmation-v1"
            or aggregate.get("state") != "completed" or aggregate.get("failed")
            or aggregate.get("pending") or aggregate.get("locked_test_accessed") is not False):
        raise ValueError(f"matching F5-G confirmation controls are not complete and clean: {path}")
    summaries = aggregate.get("summaries")
    if not isinstance(summaries, dict):
        raise ValueError("F5-G confirmation aggregate has no summaries")
    selected: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256(path.read_bytes())
    root = path.parent
    for fold in folds:
        for seed in seeds:
            run_id = f"F5-G-fold{fold}-seed{seed}"
            row = summaries.get(run_id)
            if not isinstance(row, dict) or row.get("identity") is None:
                raise ValueError(f"missing matching F5-G control summary: {run_id}")
            run_dir = root / run_id
            config = _read_json(run_dir / "config.json")
            status = _read_json(run_dir / "training-status.json")
            checkpoint = _read_json(run_dir / "checkpoints/checkpoint.json")
            identity = row["identity"]
            status_summary = status.get("confirmation_summary") or {}
            if (config.get("identity") != identity or config.get("model") != "F5-G"
                    or int(config.get("fold", -1)) != fold or int(config.get("seed", -1)) != seed
                    or status.get("identity") != identity or status.get("state") != "completed"
                    or checkpoint.get("identity") != identity
                    or config.get("locked_test_accessed") is not False
                    or status.get("locked_test_accessed") is not False
                    or status_summary.get("identity") != identity
                    or status_summary.get("best_metrics") != row.get("best_metrics")
                    or not (run_dir / "checkpoints/best.pt").is_file()
                    or not (run_dir / "checkpoints/last.pt").is_file()):
                raise ValueError(f"F5-G control artifacts are inconsistent: {run_id}")
            _assert_finite(row.get("best_metrics", {}), f"control.{run_id}.metrics")
            selected[run_id] = row
            selected[run_id] = {
                **row,
                "checkpoint_sha256": sha256_file(run_dir / "checkpoints/best.pt"),
                "control_config_sha256": sha256_file(run_dir / "config.json"),
            }
            for artifact in (run_dir / "config.json", run_dir / "training-status.json",
                             run_dir / "checkpoints/checkpoint.json"):
                digest.update(artifact.read_bytes())
    return selected, digest.hexdigest()


def _comparison(candidate: dict[str, Any], controls: dict[str, dict[str, Any]],
                baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for model_id in MODELS:
        rows = [(key, row) for key, row in candidate.items() if row.get("model") == model_id]
        by_model[model_id] = {
            "n_runs": len(rows),
            "macro_metrics": {metric: _mean_std([
                _metric_values(row).get(metric) for _, row in rows
                if _metric_values(row).get(metric) is not None]) for metric in METRICS},
            "per_dataset": {dataset: {metric: _mean_std([
                _metric_values(row, dataset=dataset).get(metric) for _, row in rows
                if _metric_values(row, dataset=dataset).get(metric) is not None])
                for metric in METRICS} for dataset in DATASETS},
        }
        refs: dict[str, Any] = {}
        for ref_name, source in (("F5-G", controls), ("F2", baselines), ("F4", baselines)):
            pairs = []
            for run_id, row in rows:
                reference_id = f"{ref_name}-fold{row['fold']}-seed{row['seed']}"
                reference = source.get(reference_id)
                if reference is None:
                    continue
                left, right = _metric_values(row), _metric_values(reference)
                delta = {metric: left[metric] - right[metric]
                         for metric in METRICS if metric in left and metric in right}
                per_ds = {}
                for dataset in DATASETS:
                    lds = _metric_values(row, dataset=dataset)
                    rds = _metric_values(reference, dataset=dataset)
                    per_ds[dataset] = {metric: lds[metric] - rds[metric]
                                       for metric in METRICS if metric in lds and metric in rds}
                pairs.append({"run_id": run_id, "reference_run_id": reference_id,
                              "delta": delta, "per_dataset_delta": per_ds})
            metrics = {}
            for metric in METRICS:
                vals = [pair["delta"][metric] for pair in pairs if metric in pair["delta"]]
                stats = _mean_std(vals)
                stats["improved_pairs"] = sum(
                    value < 0 if metric in {"mae", "rmse"} else value > 0 for value in vals)
                metrics[metric] = stats
            refs[f"vs_{ref_name}"] = {"n_pairs": len(pairs), "metrics": metrics, "pairs": pairs}
        paired[model_id] = refs
    return {"by_model": by_model, "paired_vs": paired}


def _decision_gates(analysis: dict[str, Any]) -> dict[str, Any]:
    outcomes = {}
    for model_id, references in analysis["paired_vs"].items():
        f4 = references["vs_F4"]["metrics"]
        g = references["vs_F5-G"]["metrics"]
        tad_pairs = references["vs_F4"]["pairs"]
        tad_deltas = [pair["per_dataset_delta"].get("tad66k", {}).get("spearman")
                      for pair in tad_pairs]
        tad_deltas = [value for value in tad_deltas if value is not None]
        fold_spearman: dict[int, list[float]] = {}
        for pair in references["vs_F5-G"]["pairs"]:
            fold_spearman.setdefault(int(pair["fold"]) if "fold" in pair else
                                     int(pair["run_id"].split("-fold")[1].split("-")[0]), []).append(
                                         pair["delta"].get("spearman", float("nan")))
        by_fold = {str(fold): sum(values) / len(values) for fold, values in fold_spearman.items() if values}
        checks = {
            "all_four_matched_runs_present": len(references["vs_F5-G"]["pairs"]) == 4,
            "macro_spearman_vs_F4": (len(references["vs_F4"]["pairs"]) == 4
                                     and f4["spearman"]["mean"] is not None
                                     and f4["spearman"]["mean"] >= GATES["macro_spearman_vs_F4_min"]),
            "tad66k_spearman_vs_F4": len(tad_deltas) == 4 and
                sum(tad_deltas) / len(tad_deltas) >
                GATES["tad66k_spearman_delta_vs_F4_min_exclusive"],
            "top5_lift_vs_G": (len(references["vs_F5-G"]["pairs"]) == 4
                               and g["top5_lift"]["mean"] is not None
                               and g["top5_lift"]["mean"] >= GATES["top5_lift_delta_vs_G_min"]),
            "macro_mae_vs_G": (len(references["vs_F5-G"]["pairs"]) == 4
                               and g["mae"]["mean"] is not None
                               and g["mae"]["mean"] <= GATES["macro_mae_delta_vs_G_max"]),
            "both_folds_improve_spearman_vs_G": len(by_fold) == 2 and all(
                len(fold_spearman[fold]) == 2 for fold in fold_spearman
            ) and all(
                value > GATES["each_diagnostic_fold_spearman_delta_vs_G_min"]
                for value in by_fold.values()),
        }
        outcomes[model_id] = {"checks": checks, "eligible": all(checks.values()),
                              "fold_spearman_delta_vs_G": by_fold}
    return {"predeclared_thresholds": GATES, "candidate_outcomes": outcomes,
            "locked_test_accessed": False}


def _write_status(path: Path, *, state: str, specs: list[RunSpec],
                  completed: list[str], failed: list[dict[str, Any]],
                  current: str | None, started_at: float) -> dict[str, Any]:
    done = set(completed) | {row["run_id"] for row in failed}
    value = {
        "experiment_version": VERSION, "state": state,
        "started_at_unix": started_at, "updated_at_unix": time.time(),
        "current_run": current, "total": len(specs), "completed": list(completed),
        "failed": list(failed),
        "pending": [spec.run_id for spec in specs if spec.run_id not in done],
        "locked_test_accessed": False, "technical_auxiliary": False,
        "signal_dropout": False,
    }
    _atomic_json(path, value)
    return value


def _aggregate(specs: list[RunSpec], completed: list[str], failed: list[dict[str, Any]],
               summaries: dict[str, Any], *, state: str, controls: dict[str, Any],
               baselines: dict[str, Any], control_digest: str,
               baseline_digest: str, started_at: float) -> dict[str, Any]:
    analysis = _comparison(summaries, controls, baselines)
    return {
        "experiment_version": VERSION, "state": state, "total": len(specs),
        "completed": list(completed), "failed": list(failed),
        "pending": [spec.run_id for spec in specs
                    if spec.run_id not in set(completed) | {row["run_id"] for row in failed}],
        "summaries": summaries, "paired_analysis": analysis,
        "decision_gates": _decision_gates(analysis),
        "control_reference_sha256": control_digest,
        "baseline_reference_sha256": baseline_digest,
        "started_at_unix": started_at,
        "elapsed_seconds": max(0.0, time.time() - started_at),
        "locked_test_accessed": False, "technical_auxiliary": False,
        "signal_dropout": False, "updated_at_unix": time.time(),
    }


def plan_matrix(args: argparse.Namespace) -> dict[str, Any]:
    specs = ablation_specs()
    controls, _ = _load_g_controls(Path(args.control_root) / "aggregate-metrics.json")
    planned = []
    for spec in specs:
        paths = artifact_paths(args.data_root, args.baseline_root, spec)
        model = _make_model(spec.model)
        planned.append({"spec": asdict(spec), "model_config": model.architecture_config(),
                        "f2_checkpoint": str(paths["f2_checkpoint"]),
                        "f5_g_control": str(Path(args.control_root) /
                                             f"F5-G-fold{spec.fold}-seed{spec.seed}"),
                        "normalization": str(paths["normalization"]),
                        "protocol": _protocol(args, spec),
                        "run_dir": str(Path(args.output_root) / spec.run_id)})
    return {
        "experiment_version": VERSION, "output_root": str(args.output_root),
        "total": len(specs), "runs": planned, "folds": list(FOLDS),
        "seeds": list(SEEDS), "models": list(MODELS),
        "controls_reused": len(controls), "locked_test_accessed": False,
        "technical_auxiliary": False, "signal_dropout": False,
        "predeclared_gates": GATES,
        "no_gate_capacity_note": "Same embedding branches, fusion head, and 98.5% of G trainable parameters; removes the 4,392 gate parameters.",
        "shared_training_protocol": {"device": args.device, "batch_size": args.batch_size,
                                     "num_workers": args.num_workers, "epochs": args.epochs,
                                     "patience": args.patience, "min_epochs": args.min_epochs},
        "ranking_min_normalized_target_gap_by_model": {
            model: PAIR_GAP if model == "F5-G-Pair05" else 0.0 for model in MODELS
        },
    }


def _validate_roots(args: argparse.Namespace, output_root: Path) -> None:
    _validate_roots_base(args, output_root)
    control_root = Path(args.control_root).resolve()
    if output_root == control_root or output_root in control_root.parents or control_root in output_root.parents:
        raise ValueError("ablation output must not overlap the F5-G confirmation controls")
    if output_root.name != VERSION:
        raise ValueError(f"output root must be the versioned {VERSION} directory")


def _validate_roots_base(args: argparse.Namespace, output_root: Path) -> None:
    # Reuse the screen's generic repository/data/baseline separation checks.
    _validate_screen_roots(args, output_root)


def execute_matrix(args: argparse.Namespace) -> dict[str, Any]:
    specs = ablation_specs()
    output_root = Path(args.output_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    control_root = Path(args.control_root).resolve()
    _validate_roots(args, output_root)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    controls, control_digest = _load_g_controls(control_root / "aggregate-metrics.json")
    baselines, baseline_digest = _load_baseline_reference(baseline_root / "aggregate-metrics.json")
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_status = output_root / "matrix-status.json"
    aggregate_path = output_root / "aggregate-metrics.json"
    started_at = time.time()
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    fold_inputs: dict[int, tuple[Any, Any, Any, dict[str, Any], dict[str, str]]] = {}
    _write_status(matrix_status, state="running", specs=specs, completed=completed,
                  failed=failed, current=None, started_at=started_at)
    _atomic_json(aggregate_path, _aggregate(
        specs, completed, failed, summaries, state="running", controls=controls,
        baselines=baselines, control_digest=control_digest,
        baseline_digest=baseline_digest, started_at=started_at))

    for spec in specs:
        run_dir = output_root / spec.run_id
        status_path = run_dir / "training-status.json"
        config_path = run_dir / "config.json"
        checkpoint_dir = run_dir / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        preserve_status = False
        identity = None
        initialization = None
        try:
            if spec.fold not in fold_inputs:
                fold_inputs[spec.fold] = _prepare_inputs(args, spec, baseline_root)
            train_data, validation_data, _normalizer, fold_paths, cache_identity = fold_inputs[spec.fold]
            paths = artifact_paths(args.data_root, baseline_root, spec)
            torch.manual_seed(spec.seed)
            model = _make_model(spec.model)
            initialization = initialize_from_fold_f2(
                model, paths["f2_checkpoint"], fold=spec.fold, seed=spec.seed,
                split_manifest=paths["split_manifest"],
                normalization_path=paths["normalization"], limit=None)
            calibration_ranges = _ranges(train_data)
            protocol = _protocol(args, spec)
            identity, payload = _atomic_identity(
                spec, split_sha256=sha256_file(paths["split_manifest"]),
                normalization_sha256=sha256_file(paths["normalization"]),
                cache_identity=cache_identity,
                architecture=model.architecture_config(), initialization=initialization,
                control_reference={
                    "run_id": f"F5-G-fold{spec.fold}-seed{spec.seed}",
                    "identity": controls[f"F5-G-fold{spec.fold}-seed{spec.seed}"]["identity"],
                    "checkpoint_sha256": controls[f"F5-G-fold{spec.fold}-seed{spec.seed}"]["checkpoint_sha256"],
                    "config_sha256": controls[f"F5-G-fold{spec.fold}-seed{spec.seed}"]["control_config_sha256"],
                },
                protocol=protocol)
            config = {
                "experiment_version": VERSION, "run_id": spec.run_id,
                "model": spec.model, "fold": spec.fold, "seed": spec.seed,
                "identity": identity, "identity_payload": payload,
                "model_config": model.architecture_config(),
                "f2_initialization": initialization,
                "matched_f5_g_control": f"F5-G-fold{spec.fold}-seed{spec.seed}",
                "matched_f5_g_control_identity": controls[
                    f"F5-G-fold{spec.fold}-seed{spec.seed}"]["identity"],
                "split_manifest": str(paths["split_manifest"]),
                "normalization_path": str(paths["normalization"]),
                "calibration_ranges": calibration_ranges,
                "locked_test_accessed": False, **protocol,
            }
            if reusable_completed_run(run_dir, identity):
                status = _read_json(status_path)
                summary = status.get("ablation_summary") or {
                    "run_id": spec.run_id, "model": spec.model, "fold": spec.fold,
                    "seed": spec.seed, "identity": identity,
                    "best_epoch": status.get("best_epoch"),
                    "elapsed_seconds": status.get("elapsed_seconds"),
                    "best_metrics": status.get("best_metrics", {}),
                    "diagnostics": status.get("diagnostics", {}),
                }
                summaries[spec.run_id] = summary
                completed.append(spec.run_id)
            else:
                if config_path.exists() and _read_json(config_path).get("identity") != identity:
                    preserve_status = True
                    raise ValueError(f"existing run identity mismatch; refusing overwrite: {spec.run_id}")
                prior_status = _read_json(status_path) if status_path.exists() else {}
                if prior_status.get("state") == "failed" and not args.retry_failed:
                    failed.append({"run_id": spec.run_id,
                                   "error": "prior run failed; use --retry-failed",
                                   "identity": identity, "reused_failure": True})
                else:
                    resume = None
                    checkpoint_meta = checkpoint_dir / "checkpoint.json"
                    if checkpoint_meta.is_file():
                        if _read_json(checkpoint_meta).get("identity") != identity:
                            preserve_status = True
                            raise ValueError(f"checkpoint identity mismatch: {spec.run_id}")
                        if not (checkpoint_dir / "last.pt").is_file():
                            raise ValueError(f"resume metadata lacks last.pt: {spec.run_id}")
                        resume = checkpoint_dir
                    _atomic_json(config_path, config)
                    _atomic_json(status_path, {"state": "running", "run_id": spec.run_id,
                                               "identity": identity, "locked_test_accessed": False,
                                               "attempt_started_unix": time.time()})
                    _write_status(matrix_status, state="running", specs=specs,
                                  completed=completed, failed=failed,
                                  current=spec.run_id, started_at=started_at)
                    _atomic_json(aggregate_path, _aggregate(
                        specs, completed, failed, summaries, state="running", controls=controls,
                        baselines=baselines, control_digest=control_digest,
                        baseline_digest=baseline_digest, started_at=started_at))
                    try:
                        run_started = time.monotonic()
                        result = train_fusion(
                            model, train_data, validation_data, run_dir,
                            config=config, epochs=args.epochs,
                            batch_size=args.batch_size, seed=spec.seed,
                            device=args.device, patience=args.patience,
                            min_epochs=args.min_epochs,
                            calibrators=DatasetCalibrationHeads(calibration_ranges),
                            resume=resume, num_workers=args.num_workers,
                            status_interval_batches=args.status_interval_batches,
                            ranking_min_target_gap=protocol["ranking_min_normalized_target_gap"])
                        if result.get("state") != "completed":
                            raise RuntimeError(f"trainer state is {result.get('state')}")
                        checkpoint = _read_json(checkpoint_meta)
                        if checkpoint.get("identity") != identity or not (checkpoint_dir / "best.pt").is_file():
                            raise ValueError("completed checkpoint is missing or identity-mismatched")
                        _assert_finite(result.get("best_metrics", {}), "best_metrics")
                        best_state = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
                        model.load_state_dict(best_state["model"])
                        summary = {
                            "run_id": spec.run_id, "model": spec.model,
                            "fold": spec.fold, "seed": spec.seed, "identity": identity,
                            "best_epoch": result.get("best_epoch"),
                            "elapsed_seconds": result.get("elapsed_seconds"),
                            "wall_seconds": time.monotonic() - run_started,
                            "best_metrics": result.get("best_metrics", {}),
                            "diagnostics": _candidate_diagnostics(
                                model, validation_data, args.device, args.batch_size),
                        }
                        final_status = _read_json(status_path)
                        final_status.update({"state": "completed", "identity": identity,
                                             "locked_test_accessed": False,
                                             "f2_initialization": initialization,
                                             "ablation_summary": summary})
                        _atomic_json(status_path, final_status)
                        summaries[spec.run_id] = summary
                        completed.append(spec.run_id)
                    except Exception:
                        error = traceback.format_exc()
                        failed.append({"run_id": spec.run_id, "identity": identity,
                                       "error": error, "timestamp_unix": time.time(),
                                       "retry_requires_flag": True})
                        last_metrics = None
                        if status_path.is_file():
                            try:
                                last_metrics = _read_json(status_path).get("last_metrics")
                            except (OSError, ValueError):
                                pass
                        _atomic_json(status_path, {
                            "state": "failed", "run_id": spec.run_id,
                            "identity": identity, "error": error,
                            "last_metrics": last_metrics,
                            "locked_test_accessed": False,
                            "f2_initialization": initialization,
                        })
            _write_status(matrix_status, state="running", specs=specs,
                          completed=completed, failed=failed,
                          current=spec.run_id, started_at=started_at)
            _atomic_json(aggregate_path, _aggregate(
                specs, completed, failed, summaries, state="running", controls=controls,
                baselines=baselines, control_digest=control_digest,
                baseline_digest=baseline_digest, started_at=started_at))
        except Exception:
            error = traceback.format_exc()
            failed.append({"run_id": spec.run_id, "identity": identity,
                           "error": error, "timestamp_unix": time.time(),
                           "retry_requires_flag": True})
            if run_dir.exists() and not preserve_status:
                _atomic_json(status_path, {"state": "failed", "run_id": spec.run_id,
                                           "identity": identity, "error": error,
                                           "locked_test_accessed": False,
                                           "f2_initialization": initialization})
        _write_status(matrix_status, state="running", specs=specs,
                      completed=completed, failed=failed,
                      current=spec.run_id, started_at=started_at)
        _atomic_json(aggregate_path, _aggregate(
            specs, completed, failed, summaries, state="running", controls=controls,
            baselines=baselines, control_digest=control_digest,
            baseline_digest=baseline_digest, started_at=started_at))

    state = "completed" if len(completed) == len(specs) and not failed else "completed_with_failures"
    final = _aggregate(specs, completed, failed, summaries, state=state,
                       controls=controls, baselines=baselines,
                       control_digest=control_digest, baseline_digest=baseline_digest,
                       started_at=started_at)
    _write_status(matrix_status, state=state, specs=specs, completed=completed,
                  failed=failed, current=None, started_at=started_at)
    _atomic_json(aggregate_path, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute 12 runs; default is read-only plan")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--baseline-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/baseline-matrix-v6"))
    parser.add_argument("--control-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/f5-confirmation-v1"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--status-interval-batches", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-epochs", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.output_root = args.output_root or Path(args.data_root) / f"fusion/experiments/{VERSION}"
    try:
        _validate_roots(args, args.output_root)
        result = execute_matrix(args) if args.execute else plan_matrix(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("state", "planned") in {"planned", "completed"} else 1
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
