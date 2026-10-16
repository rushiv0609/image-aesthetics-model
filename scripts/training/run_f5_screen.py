#!/usr/bin/env python3
"""Plan or execute the fixed 16-run F3C/F5 preflight and screen.

The only scheduled candidates are F3C and F5-R/G/X over folds 0 and 4 with
seeds 20260920 and 20260921. Execution reads train/validation partitions only;
there is no test-partition option. Artifacts default to external storage at
``fusion/experiments/f5-screen-v1``. Planning is the default and does not train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import DatasetCalibrationHeads, build_baseline
from image_aesthetics.training.f5_initialization import (
    initialize_from_fold_f2,
    sha256_file,
)
from image_aesthetics.training.fusion_trainer import train_fusion


FOLDS = (0, 4)
SEEDS = (20260920, 20260921)
MODELS = ("F3C", "F5-R", "F5-G", "F5-X")
F5_MODELS = frozenset(("F5-R", "F5-G", "F5-X"))
DATASETS = ("aadb", "para", "flickr_aes", "tad66k")
SCREEN_VERSION = "f5-screen-v1"
METRICS = ("spearman", "pairwise_accuracy", "top1_lift", "top5_lift",
           "top10_lift", "top1_overlap", "top5_overlap", "top10_overlap",
           "mae", "rmse")


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model: str
    fold: int
    seed: int


def screen_specs() -> list[RunSpec]:
    """Return the immutable 4-model × 2-fold × 2-seed screen (16 runs)."""
    return [RunSpec(f"{model}-fold{fold}-seed{seed}", model, fold, seed)
            for fold in FOLDS for seed in SEEDS for model in MODELS]


def artifact_paths(data_root: Path, baseline_root: Path, spec: RunSpec) -> dict[str, Any]:
    """Resolve only the frozen split, joined-ten caches, normalizer, and F2 source."""
    return {
        "split_manifest": Path(data_root) / "fusion/prepared/v6/samples.jsonl",
        "joins": {dataset: Path(data_root) / dataset / "features/joined-ten-v1"
                  for dataset in DATASETS},
        "normalization": Path(baseline_root) / "normalization" / f"fold{spec.fold}.json",
        "f2_checkpoint": (Path(baseline_root) /
                          f"F2-fold{spec.fold}-seed{spec.seed}" /
                          "checkpoints/best.pt"),
    }


def run_identity(spec: RunSpec, *, split_sha256: str,
                 normalization_sha256: str, cache_identity: dict[str, str],
                 architecture: dict[str, Any], f2_initialization: dict[str, Any] | None,
                 protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Hash every data, initialization, architecture, and training choice."""
    payload = {
        "screen_version": SCREEN_VERSION,
        "spec": asdict(spec),
        "split_manifest_sha256": split_sha256,
        "normalization_sha256": normalization_sha256,
        "joined_cache_identity": dict(sorted(cache_identity.items())),
        "architecture": architecture,
        "f2_initialization": f2_initialization,
        "protocol": protocol,
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _cache_hashes(join_dirs: dict[str, Path]) -> dict[str, str]:
    result = {}
    for dataset, path in sorted(join_dirs.items()):
        if not path.is_dir():
            raise FileNotFoundError(f"missing joined-ten cache: {path}")
        digest = hashlib.sha256()
        found = False
        for filename in ("join-audit.json", "join.jsonl"):
            candidate = path / filename
            if candidate.is_file():
                found = True
                with candidate.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        if not found:
            raise FileNotFoundError(f"cache lacks join audit/index identity: {path}")
        result[dataset] = digest.hexdigest()
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def reusable_completed_run(run_dir: Path, identity: str) -> bool:
    """Require matching config, completed status, and checkpoint metadata/files."""
    run_dir = Path(run_dir)
    try:
        config = _read_json(run_dir / "config.json")
        status = _read_json(run_dir / "training-status.json")
        checkpoint_meta = _read_json(run_dir / "checkpoints/checkpoint.json")
        return (
            config.get("identity") == identity
            and status.get("state") == "completed"
            and status.get("identity") == identity
            and checkpoint_meta.get("identity") == identity
            and (run_dir / "checkpoints/best.pt").is_file()
            and (run_dir / "checkpoints/last.pt").is_file()
        )
    except (OSError, ValueError, TypeError):
        return False


def _matrix_status(path: Path, *, state: str, specs: list[RunSpec],
                   completed: list[str], failed: list[dict[str, Any]],
                   current: str | None, started_at: float, updated_at: float | None = None):
    done = set(completed) | {entry["run_id"] for entry in failed}
    value = {
        "screen_version": SCREEN_VERSION,
        "state": state,
        "started_at_unix": started_at,
        "updated_at_unix": updated_at or time.time(),
        "current_run": current,
        "total": len(specs),
        "completed": list(completed),
        "failed": list(failed),
        "pending": [spec.run_id for spec in specs if spec.run_id not in done],
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
    }
    _atomic_json(path, value)
    return value


def _protocol(args: argparse.Namespace) -> dict[str, Any]:
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
        "technical_auxiliary": False,
        "signal_dropout": False,
        "limit": None,
    }


def _validate_roots(args: argparse.Namespace, output_root: Path) -> None:
    repo_root = ROOT.resolve()
    output_root = output_root.resolve()
    baseline_root = Path(args.baseline_root).resolve()
    data_root = Path(args.data_root).resolve()
    if output_root == repo_root or repo_root in output_root.parents:
        raise ValueError("experiment output must stay outside the Git repository")
    if (output_root == baseline_root or baseline_root in output_root.parents
            or output_root in baseline_root.parents):
        raise ValueError("experiment output must not overlap the F2/F4 baseline root")
    if output_root == data_root or output_root in data_root.parents:
        raise ValueError("experiment output must not overwrite the dataset root")
    if args.batch_size < 1 or args.num_workers < 0 or args.status_interval_batches < 1:
        raise ValueError("batch size/status interval must be positive and workers non-negative")
    if args.epochs < 1 or args.patience < 1 or args.min_epochs < 1 or args.min_epochs > args.epochs:
        raise ValueError("require epochs >= min_epochs >= 1 and patience >= 1")


def _prepare_inputs(args: argparse.Namespace, spec: RunSpec,
                    baseline_root: Path) -> tuple[Any, Any, FeatureNormalizer, dict[str, Path], dict[str, str]]:
    paths = artifact_paths(args.data_root, baseline_root, spec)
    for path in (paths["split_manifest"], paths["normalization"], *paths["joins"].values()):
        if not path.exists():
            raise FileNotFoundError(f"required screen input missing: {path}")
    train_base = FusionTrainingDataset(
        paths["split_manifest"], paths["joins"], partition="train", validation_fold=spec.fold)
    validation_base = FusionTrainingDataset(
        paths["split_manifest"], paths["joins"], partition="validation", validation_fold=spec.fold)
    expected = train_base.normalization_identity()
    normalizer = FeatureNormalizer.from_file(paths["normalization"], expected_identity=expected)
    train_data = train_base.normalized(normalizer)
    validation_data = validation_base.normalized(normalizer)
    cache_identity = _cache_hashes(paths["joins"])
    return train_data, validation_data, normalizer, paths, cache_identity


def _metric_report(summary: dict[str, Any]) -> dict[str, Any]:
    report = summary.get("best_metrics", summary)
    return report.get("val", report)


def _assert_finite(value: Any, path: str = "metrics") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_finite(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_finite(nested, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite metric value at {path}: {value}")


def _metric_values(summary: dict[str, Any], *, dataset: str | None = None) -> dict[str, float]:
    report = _metric_report(summary)
    source = report.get("per_dataset", {}).get(dataset, {}) if dataset else report.get("macro", {})
    return {metric: float(source[metric]) for metric in METRICS if metric in source}


def _mean_std(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None}
    return {"n": len(values), "mean": statistics.mean(values),
            "sd": statistics.pstdev(values) if len(values) > 1 else 0.0}


def paired_analysis(screen_summaries: dict[str, dict[str, Any]],
                    baseline_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-model metrics and exact fold/seed paired F2/F4 deltas."""
    model_aggregates: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for model in MODELS:
        candidates = [(run_id, row) for run_id, row in screen_summaries.items()
                      if run_id.startswith(model + "-")]
        metric_stats = {}
        for metric in METRICS:
            values = [_metric_values(row).get(metric) for _, row in candidates]
            metric_stats[metric] = _mean_std([value for value in values if value is not None])
        per_dataset = {}
        for dataset in DATASETS:
            per_dataset[dataset] = {
                metric: _mean_std([
                    value for _, row in candidates
                    if (value := _metric_values(row, dataset=dataset).get(metric)) is not None
                ]) for metric in METRICS
            }
        model_aggregates[model] = {
            "n_runs": len(candidates), "macro_metrics": metric_stats,
            "per_dataset": per_dataset,
            "stability": {
                "folds_present": sorted({int(row["fold"]) for _, row in candidates}),
                "seeds_present": sorted({int(row["seed"]) for _, row in candidates}),
            },
        }

        model_paired = {}
        for reference in ("F2", "F4"):
            pairs = []
            for run_id, row in candidates:
                fold, seed = int(row["fold"]), int(row["seed"])
                reference_id = f"{reference}-fold{fold}-seed{seed}"
                baseline = baseline_summaries.get(reference_id)
                if baseline is None:
                    continue
                candidate_values = _metric_values(row)
                reference_values = _metric_values(baseline)
                deltas = {metric: candidate_values[metric] - reference_values[metric]
                          for metric in METRICS
                          if metric in candidate_values and metric in reference_values}
                per_ds_delta = {}
                for dataset in DATASETS:
                    left = _metric_values(row, dataset=dataset)
                    right = _metric_values(baseline, dataset=dataset)
                    per_ds_delta[dataset] = {
                        metric: left[metric] - right[metric]
                        for metric in METRICS if metric in left and metric in right
                    }
                pairs.append({"run_id": run_id, "reference_run_id": reference_id,
                              "delta": deltas, "per_dataset_delta": per_ds_delta})
            by_metric = {}
            for metric in METRICS:
                deltas = [pair["delta"][metric] for pair in pairs if metric in pair["delta"]]
                stats = _mean_std(deltas)
                stats["improved_pairs"] = sum(
                    delta < 0 if metric in {"mae", "rmse"} else delta > 0
                    for delta in deltas)
                by_metric[metric] = stats
            paired_per_dataset = {
                dataset: {
                    metric: _mean_std([
                        value for pair in pairs
                        if (value := pair["per_dataset_delta"][dataset].get(metric)) is not None
                    ]) for metric in METRICS
                } for dataset in DATASETS
            }
            model_paired[f"vs_{reference}"] = {
                "n_pairs": len(pairs), "metrics": by_metric,
                "per_dataset": paired_per_dataset, "pairs": pairs,
            }
        paired[model] = model_paired
    return {"by_model": model_aggregates, "paired_vs_baseline": paired}


def _load_baseline_reference(path: Path) -> dict[str, dict[str, Any]]:
    aggregate = _read_json(path)
    if aggregate.get("state") != "completed" or aggregate.get("failed") or aggregate.get("pending"):
        raise ValueError(f"baseline reference is not a fully completed matrix: {path}")
    summaries = aggregate.get("summaries")
    if not isinstance(summaries, dict):
        raise ValueError("baseline aggregate has no summaries mapping")
    required = [f"{model}-fold{fold}-seed{seed}"
                for model in ("F2", "F4") for fold in FOLDS for seed in SEEDS]
    missing = [run_id for run_id in required if run_id not in summaries]
    if missing:
        raise ValueError(f"baseline reference lacks paired F2/F4 runs: {missing}")
    return summaries


def _candidate_diagnostics(model: torch.nn.Module, validation_data: Any,
                           device: str, batch_size: int) -> dict[str, Any]:
    model.eval()
    residuals = []
    gates = []
    with torch.no_grad():
        for batch in DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0):
            features = batch["features"].to(device=device, dtype=torch.float32)
            if hasattr(model, "bounded_residual"):
                residuals.append(model.bounded_residual(features).detach())
            if hasattr(model, "gate_values"):
                gates.append(model.gate_values(features).detach())
    result: dict[str, Any] = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    if residuals:
        values = torch.cat(residuals).float()
        result["residual"] = {
            "mean": float(values.mean().cpu()), "sd": float(values.std(unbiased=False).cpu()),
            "mean_abs": float(values.abs().mean().cpu()),
            "max_abs": float(values.abs().max().cpu()),
            "bound": float(model.residual_bound),
        }
    if gates:
        values = torch.cat(gates).float()
        result["gate"] = {
            "mean": float(values.mean().cpu()), "sd": float(values.std(unbiased=False).cpu()),
            "min": float(values.min().cpu()), "max": float(values.max().cpu()),
            "fraction_near_bound": float((values.sub(1).abs() >= 0.225).float().mean().cpu()),
        }
    return result


def _ranges(dataset: Any) -> dict[str, tuple[float, float]]:
    source = dataset.dataset if hasattr(dataset, "dataset") else dataset
    result: dict[str, tuple[float, float]] = {}
    for dataset_id, _index, row, _target in source.samples:
        bounds = tuple(map(float, row["native_score_range"]))
        if dataset_id in result and result[dataset_id] != bounds:
            raise ValueError(f"inconsistent native score range for {dataset_id}")
        result[dataset_id] = bounds
    return result


def build_run_config(args: argparse.Namespace, spec: RunSpec,
                     split_manifest: Path, normalization_path: Path,
                     cache_identity: dict[str, str], model: torch.nn.Module,
                     f2_initialization: dict[str, Any] | None,
                     calibration_ranges: dict[str, tuple[float, float]]) -> dict[str, Any]:
    architecture = model.architecture_config()
    identity, payload = run_identity(
        spec,
        split_sha256=sha256_file(split_manifest),
        normalization_sha256=sha256_file(normalization_path),
        cache_identity=cache_identity,
        architecture=architecture,
        f2_initialization=f2_initialization,
        protocol=_protocol(args),
    )
    return {
        "run_id": spec.run_id,
        "model": spec.model,
        "fold": spec.fold,
        "seed": spec.seed,
        "identity": identity,
        "identity_payload": payload,
        "model_config": architecture,
        "f2_initialization": f2_initialization,
        "split_manifest": str(split_manifest),
        "normalization_path": str(normalization_path),
        "calibration_ranges": calibration_ranges,
        "locked_test_accessed": False,
        **_protocol(args),
    }


def plan_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Read-only plan of exact run identities and their expected artifacts."""
    specs = screen_specs()
    data_root = Path(args.data_root)
    baseline_root = Path(args.baseline_root)
    plans = []
    for spec in specs:
        paths = artifact_paths(data_root, baseline_root, spec)
        model = build_baseline(spec.model)
        config_stub = {
            "model_config": model.architecture_config(),
            "f2_checkpoint": str(paths["f2_checkpoint"]) if spec.model in F5_MODELS else None,
            "normalization": str(paths["normalization"]),
            "run_dir": str(Path(args.output_root) / spec.run_id),
        }
        plans.append({"spec": asdict(spec), **config_stub})
    return {
        "screen_version": SCREEN_VERSION,
        "output_root": str(Path(args.output_root)),
        "split_manifest": str(Path(args.data_root) / "fusion/prepared/v6/samples.jsonl"),
        "total": len(specs),
        "runs": plans,
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
    }


def execute_screen(args: argparse.Namespace) -> dict[str, Any]:
    specs = screen_specs()
    output_root = Path(args.output_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    _validate_roots(args, output_root)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    if args.device == "cpu" and torch.backends.mps.is_available():
        # CPU remains a supported explicit override for diagnostics, but cannot
        # be mistaken for the approved default MPS screen.
        pass
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "matrix-status.json"
    aggregate_path = output_root / "aggregate-metrics.json"
    baseline_aggregate_path = baseline_root / "aggregate-metrics.json"
    baseline_summaries = _load_baseline_reference(baseline_aggregate_path)
    baseline_identity = sha256_file(baseline_aggregate_path)
    started_at = time.time()
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    fold_inputs: dict[int, tuple[Any, Any, FeatureNormalizer, dict[str, Path], dict[str, str]]] = {}

    _matrix_status(status_path, state="running", specs=specs, completed=completed,
                   failed=failed, current=None, started_at=started_at)
    _atomic_json(aggregate_path, {
        "screen_version": SCREEN_VERSION, "state": "running", "total": len(specs),
        "completed": completed, "failed": failed, "pending": [s.run_id for s in specs],
        "summaries": summaries,
        "paired_analysis": paired_analysis(summaries, baseline_summaries),
        "baseline_reference_sha256": baseline_identity,
        "locked_test_accessed": False,
    })

    for spec in specs:
        run_dir = output_root / spec.run_id
        config_path = run_dir / "config.json"
        training_status_path = run_dir / "training-status.json"
        checkpoint_dir = run_dir / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        preserve_existing_status = False
        try:
            if spec.fold not in fold_inputs:
                fold_inputs[spec.fold] = _prepare_inputs(args, spec, baseline_root)
            train_data, validation_data, _normalizer, paths, cache_identity = fold_inputs[spec.fold]
            # Seed parameter construction before instantiating each candidate;
            # train_fusion reseeds the same run seed for its data/training RNG.
            torch.manual_seed(spec.seed)
            model = build_baseline(spec.model)
            initialization = None
            if spec.model in F5_MODELS:
                initialization = initialize_from_fold_f2(
                    model, paths["f2_checkpoint"], fold=spec.fold, seed=spec.seed,
                    split_manifest=paths["split_manifest"],
                    normalization_path=paths["normalization"], limit=None)
            calibration_ranges = _ranges(train_data)
            config = build_run_config(
                args, spec, paths["split_manifest"], paths["normalization"],
                cache_identity, model, initialization, calibration_ranges)
            identity = config["identity"]

            if reusable_completed_run(run_dir, identity):
                status = _read_json(training_status_path)
                summary = status.get("screen_summary") or {
                    "run_id": spec.run_id, "model": spec.model, "fold": spec.fold,
                    "seed": spec.seed, "identity": identity,
                    "best_epoch": status.get("best_epoch"),
                    "elapsed_seconds": status.get("elapsed_seconds"),
                    "best_metrics": status.get("best_metrics") or {},
                    "diagnostics": status.get("diagnostics", {}),
                }
                completed.append(spec.run_id)
                summaries[spec.run_id] = summary
                _matrix_status(status_path, state="running", specs=specs,
                               completed=completed, failed=failed,
                               current=spec.run_id, started_at=started_at)
                _atomic_json(aggregate_path, _aggregate_payload(
                    specs, completed, failed, summaries, state="running",
                    baseline_summaries=baseline_summaries,
                    baseline_identity=baseline_identity,
                    started_at_unix=started_at))
                continue

            if config_path.exists():
                previous_config = _read_json(config_path)
                previous_identity = previous_config.get("identity")
                if previous_identity != identity:
                    preserve_existing_status = True
                    raise ValueError(
                        f"existing run identity mismatch for {spec.run_id}; refusing overwrite")
            prior_run_status = {}
            if training_status_path.exists():
                try:
                    prior_run_status = _read_json(training_status_path)
                except (OSError, ValueError):
                    prior_run_status = {}
            if prior_run_status.get("state") == "failed" and not args.retry_failed:
                failed.append({"run_id": spec.run_id,
                               "error": "previous attempt failed; pass --retry-failed to resume/retry",
                               "identity": identity, "reused_failure": True})
                _matrix_status(status_path, state="running", specs=specs,
                               completed=completed, failed=failed, current=spec.run_id,
                               started_at=started_at)
                _atomic_json(aggregate_path, _aggregate_payload(
                    specs, completed, failed, summaries, state="running",
                    baseline_summaries=baseline_summaries,
                    baseline_identity=baseline_identity,
                    started_at_unix=started_at))
                continue

            resume_dir = None
            if (checkpoint_dir / "checkpoint.json").is_file():
                metadata = _read_json(checkpoint_dir / "checkpoint.json")
                if metadata.get("identity") != identity:
                    preserve_existing_status = True
                    raise ValueError(f"checkpoint identity mismatch for {spec.run_id}")
                if not (checkpoint_dir / "last.pt").is_file():
                    raise ValueError(f"resume metadata lacks last.pt for {spec.run_id}")
                resume_dir = checkpoint_dir
            _atomic_json(config_path, config)
            _atomic_json(training_status_path, {
                "state": "running", "run_id": spec.run_id, "identity": identity,
                "attempt_started_unix": time.time(), "locked_test_accessed": False,
            })
            _matrix_status(status_path, state="running", specs=specs,
                           completed=completed, failed=failed, current=spec.run_id,
                           started_at=started_at)
            _atomic_json(aggregate_path, _aggregate_payload(
                specs, completed, failed, summaries, state="running",
                baseline_summaries=baseline_summaries,
                baseline_identity=baseline_identity,
                started_at_unix=started_at))

            log_out = run_dir / "stdout.log"
            log_err = run_dir / "stderr.log"
            with log_out.open("a") as stdout, log_err.open("a") as stderr:
                print(f"RUN_BEGIN {spec.run_id} identity={identity} resume={resume_dir is not None}",
                      file=stdout, flush=True)
                try:
                    run_started = time.monotonic()
                    calibrators = DatasetCalibrationHeads(calibration_ranges)
                    result = train_fusion(
                        model, train_data, validation_data, run_dir, config=config,
                        epochs=args.epochs, batch_size=args.batch_size, seed=spec.seed,
                        device=args.device, patience=args.patience,
                        min_epochs=args.min_epochs, calibrators=calibrators,
                        resume=resume_dir,
                        num_workers=args.num_workers,
                        status_interval_batches=args.status_interval_batches,
                    )
                    if result.get("state") != "completed":
                        raise RuntimeError(f"trainer returned non-completed state: {result.get('state')}")
                    checkpoint_metadata = _read_json(checkpoint_dir / "checkpoint.json")
                    if checkpoint_metadata.get("identity") != identity:
                        raise ValueError("completed checkpoint identity does not match planned run")
                    if not (checkpoint_dir / "best.pt").is_file():
                        raise FileNotFoundError("trainer completed without best.pt")
                    _assert_finite(result.get("best_metrics", {}), "best_metrics")
                    best_state = torch.load(checkpoint_dir / "best.pt", map_location="cpu",
                                            weights_only=False)
                    model.load_state_dict(best_state["model"])
                    diagnostics = _candidate_diagnostics(
                        model, validation_data, args.device, args.batch_size)
                    source_f2_elapsed = None
                    if initialization is not None:
                        f2_status_path = paths["f2_checkpoint"].parent.parent / "training-status.json"
                        if f2_status_path.is_file():
                            source_f2_elapsed = _read_json(f2_status_path).get("elapsed_seconds")
                    summary = {
                        "run_id": spec.run_id, "model": spec.model,
                        "fold": spec.fold, "seed": spec.seed, "identity": identity,
                        "best_epoch": result.get("best_epoch"),
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "screen_wall_seconds": time.monotonic() - run_started,
                        "source_f2_elapsed_seconds": source_f2_elapsed,
                        "best_metrics": result.get("best_metrics", {}),
                        "diagnostics": diagnostics,
                    }
                    # The trainer's status intentionally stores full validation
                    # metrics; attach run/provenance for resumable aggregation.
                    final_status = _read_json(training_status_path)
                    final_status.update({
                        "run_id": spec.run_id, "identity": identity,
                        "locked_test_accessed": False,
                        "f2_initialization": initialization,
                        "screen_summary": summary,
                    })
                    _atomic_json(training_status_path, final_status)
                    summaries[spec.run_id] = summary
                    completed.append(spec.run_id)
                    print(f"RUN_COMPLETE {spec.run_id}", file=stdout, flush=True)
                except Exception:
                    error = traceback.format_exc()
                    print(error, file=stderr, flush=True)
                    failed.append({"run_id": spec.run_id, "identity": identity,
                                   "error": error, "timestamp_unix": time.time(),
                                   "retry_requires_flag": True})
                    prior_metrics = None
                    if training_status_path.exists():
                        try:
                            prior_metrics = _read_json(training_status_path).get("last_metrics")
                        except (OSError, ValueError):
                            pass
                    _atomic_json(training_status_path, {
                        "state": "failed", "run_id": spec.run_id,
                        "identity": identity, "error": error,
                        "last_metrics": prior_metrics,
                        "locked_test_accessed": False,
                        "f2_initialization": initialization,
                    })

        except Exception:
            error = traceback.format_exc()
            failed.append({"run_id": spec.run_id, "error": error,
                           "timestamp_unix": time.time(), "retry_requires_flag": True})
            if run_dir.exists() and not preserve_existing_status:
                _atomic_json(training_status_path, {
                    "state": "failed", "run_id": spec.run_id, "error": error,
                    "locked_test_accessed": False,
                })

        _matrix_status(status_path, state="running", specs=specs,
                       completed=completed, failed=failed, current=spec.run_id,
                       started_at=started_at)
        _atomic_json(aggregate_path, _aggregate_payload(
            specs, completed, failed, summaries, state="running",
            baseline_summaries=baseline_summaries,
            baseline_identity=baseline_identity,
            started_at_unix=started_at))

    final_state = "completed" if not failed and len(completed) == len(specs) else "completed_with_failures"
    final = _aggregate_payload(specs, completed, failed, summaries, state=final_state,
                               baseline_summaries=baseline_summaries,
                               baseline_identity=baseline_identity,
                               started_at_unix=started_at)
    _matrix_status(status_path, state=final_state, specs=specs,
                   completed=completed, failed=failed, current=None,
                   started_at=started_at)
    _atomic_json(aggregate_path, final)
    return final


def _aggregate_payload(specs: list[RunSpec], completed: list[str],
                       failed: list[dict[str, Any]], summaries: dict[str, Any],
                       *, state: str,
                       baseline_summaries: dict[str, dict[str, Any]],
                       baseline_identity: str,
                       started_at_unix: float) -> dict[str, Any]:
    done = set(completed) | {entry["run_id"] for entry in failed}
    return {
        "screen_version": SCREEN_VERSION,
        "state": state,
        "total": len(specs),
        "completed": list(completed),
        "failed": list(failed),
        "pending": [spec.run_id for spec in specs if spec.run_id not in done],
        "summaries": summaries,
        "paired_analysis": paired_analysis(summaries, baseline_summaries),
        "baseline_reference_sha256": baseline_identity,
        "started_at_unix": started_at_unix,
        "elapsed_seconds": max(0.0, time.time() - started_at_unix),
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
        "updated_at_unix": time.time(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute all 16 fixed runs; default is plan-only")
    parser.add_argument("--retry-failed", action="store_true",
                        help="explicitly retry failed runs with matching identity/checkpoint")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--baseline-root", type=Path,
                        default=Path("/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/baseline-matrix-v6"))
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
    args.output_root = args.output_root or Path(args.data_root) / "fusion/experiments/f5-screen-v1"
    try:
        _validate_roots(args, args.output_root)
        if args.execute:
            result = execute_screen(args)
        else:
            result = plan_matrix(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("state", "planned") in {"planned", "completed"} else 1
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
