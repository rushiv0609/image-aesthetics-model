#!/usr/bin/env python3
"""Plan or execute the fixed 30-run F5-G/F5-X confirmation matrix.

Execution reads train/validation partitions only. There is deliberately no
locked-test partition option. Artifacts default to the external, versioned
``fusion/experiments/f5-confirmation-v1`` root; planning is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import DatasetCalibrationHeads, build_baseline
from image_aesthetics.training.f5_initialization import (
    initialize_from_fold_f2,
    sha256_file,
)
from image_aesthetics.training.fusion_trainer import train_fusion
from scripts.training.run_f5_screen import (
    METRICS,
    RunSpec,
    _assert_finite,
    _atomic_json,
    _candidate_diagnostics,
    _mean_std,
    _prepare_inputs,
    _ranges,
    _read_json,
    _validate_roots,
    artifact_paths,
    paired_analysis,
    reusable_completed_run,
)
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (20260920, 20260921, 20260922)
MODELS = ("F5-G", "F5-X")
VERSION = "f5-confirmation-v1"


def _write_matrix_status(path: Path, *, state: str, specs: list[RunSpec],
                         completed: list[str], failed: list[dict[str, Any]],
                         current: str | None, started_at: float) -> dict[str, Any]:
    done = set(completed) | {entry["run_id"] for entry in failed}
    value = {
        "experiment_version": VERSION,
        "state": state,
        "started_at_unix": started_at,
        "updated_at_unix": time.time(),
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


def _prior_failure_blocks_retry(previous_status: dict[str, Any], retry_failed: bool) -> bool:
    return previous_status.get("state") == "failed" and not retry_failed


def confirmation_specs() -> list[RunSpec]:
    """Return the fixed two-model × five-fold × three-seed matrix."""
    return [RunSpec(f"{model}-fold{fold}-seed{seed}", model, fold, seed)
            for fold in FOLDS for seed in SEEDS for model in MODELS]


def _identity(spec: RunSpec, *, split_sha256: str,
              normalization_sha256: str, cache_identity: dict[str, str],
              architecture: dict[str, Any], initialization: dict[str, Any],
              protocol: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {
        "experiment_version": VERSION,
        "spec": asdict(spec),
        "split_manifest_sha256": split_sha256,
        "normalization_sha256": normalization_sha256,
        "joined_cache_identity": dict(sorted(cache_identity.items())),
        "architecture": architecture,
        "f2_initialization": initialization,
        "protocol": protocol,
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         default=str).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


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


def _run_config(args: argparse.Namespace, spec: RunSpec, paths: dict[str, Any],
                cache_identity: dict[str, str], model: torch.nn.Module,
                initialization: dict[str, Any],
                calibration_ranges: dict[str, tuple[float, float]]) -> dict[str, Any]:
    architecture = model.architecture_config()
    protocol = _protocol(args)
    identity, payload = _identity(
        spec,
        split_sha256=sha256_file(paths["split_manifest"]),
        normalization_sha256=sha256_file(paths["normalization"]),
        cache_identity=cache_identity,
        architecture=architecture,
        initialization=initialization,
        protocol=protocol,
    )
    return {
        "experiment_version": VERSION,
        "run_id": spec.run_id,
        "model": spec.model,
        "fold": spec.fold,
        "seed": spec.seed,
        "identity": identity,
        "identity_payload": payload,
        "model_config": architecture,
        "f2_initialization": initialization,
        "split_manifest": str(paths["split_manifest"]),
        "normalization_path": str(paths["normalization"]),
        "calibration_ranges": calibration_ranges,
        "locked_test_accessed": False,
        **protocol,
    }


def _metric_report(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_metrics", summary)
    return best.get("val", best)


def _metric_values(summary: dict[str, Any]) -> dict[str, float]:
    macro = _metric_report(summary).get("macro", {})
    return {metric: float(macro[metric]) for metric in METRICS if metric in macro}


def _load_baseline_reference(baseline_root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    """Read and fingerprint the completed seed-specific F2/F4 artifacts.

    The baseline directory's legacy matrix aggregate is fold-level; the actual
    seed-paired references live in each run directory. Validate their configs,
    statuses, and checkpoint identities directly so pairings cannot silently
    collapse to fold-only summaries.
    """
    baseline_root = Path(baseline_root)
    summaries: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for model in ("F2", "F4"):
        for fold in FOLDS:
            for seed in SEEDS:
                run_id = f"{model}-fold{fold}-seed{seed}"
                run_dir = baseline_root / run_id
                config_path = run_dir / "config.json"
                status_path = run_dir / "training-status.json"
                checkpoint_path = run_dir / "checkpoints/checkpoint.json"
                config = _read_json(config_path)
                status = _read_json(status_path)
                checkpoint = _read_json(checkpoint_path)
                if (config.get("run_id") != run_id or config.get("model") != model
                        or int(config.get("fold", -1)) != fold
                        or int(config.get("seed", -1)) != seed):
                    raise ValueError(f"baseline config does not match expected run: {run_id}")
                identity = config.get("identity")
                if (not identity or status.get("state") != "completed"
                        or status.get("identity") not in (None, identity)
                        or checkpoint.get("identity") != identity
                        or not (run_dir / "checkpoints/best.pt").is_file()
                        or not (run_dir / "checkpoints/last.pt").is_file()):
                    raise ValueError(f"baseline run is incomplete or inconsistent: {run_id}")
                if config.get("locked_test_accessed") is not False:
                    raise ValueError(f"baseline run lacks explicit locked-test exclusion: {run_id}")
                metrics = status.get("best_metrics")
                if not isinstance(metrics, dict) or not metrics.get("val"):
                    raise ValueError(f"baseline run lacks validation metrics: {run_id}")
                _assert_finite(metrics, f"baseline.{run_id}.best_metrics")
                summaries[run_id] = {
                    "run_id": run_id, "model": model, "fold": fold, "seed": seed,
                    "identity": identity, "best_metrics": metrics,
                }
                for artifact in (config_path, status_path, checkpoint_path):
                    digest.update(artifact.name.encode())
                    digest.update(artifact.read_bytes())
    return summaries, digest.hexdigest()


def _aggregate(specs: list[RunSpec], completed: list[str], failed: list[dict[str, Any]],
               summaries: dict[str, Any], *, state: str,
               baseline_summaries: dict[str, dict[str, Any]],
               baseline_identity: str, started_at: float) -> dict[str, Any]:
    done = set(completed) | {entry["run_id"] for entry in failed}
    candidates = {key: value for key, value in summaries.items()
                  if value.get("model") in MODELS}
    baseline_specs = {
        key: value for key, value in baseline_summaries.items()
        if value.get("model") in {"F2", "F4"}
    }
    return {
        "experiment_version": VERSION,
        "state": state,
        "total": len(specs),
        "completed": list(completed),
        "failed": list(failed),
        "pending": [spec.run_id for spec in specs if spec.run_id not in done],
        "summaries": summaries,
        "paired_analysis": paired_analysis(candidates, baseline_specs),
        "baseline_reference_sha256": baseline_identity,
        "started_at_unix": started_at,
        "elapsed_seconds": max(0.0, time.time() - started_at),
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
        "updated_at_unix": time.time(),
    }


def plan_matrix(args: argparse.Namespace) -> dict[str, Any]:
    specs = confirmation_specs()
    baseline_root = Path(args.baseline_root)
    output_root = Path(args.output_root)
    planned = []
    for spec in specs:
        paths = artifact_paths(args.data_root, baseline_root, spec)
        model = build_baseline(spec.model)
        planned.append({
            "spec": asdict(spec),
            "model_config": model.architecture_config(),
            "f2_checkpoint": str(paths["f2_checkpoint"]),
            "normalization": str(paths["normalization"]),
            "run_dir": str(output_root / spec.run_id),
        })
    return {
        "experiment_version": VERSION,
        "output_root": str(output_root),
        "split_manifest": str(Path(args.data_root) / "fusion/prepared/v6/samples.jsonl"),
        "total": len(specs),
        "runs": planned,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "locked_test_accessed": False,
        "technical_auxiliary": False,
        "signal_dropout": False,
        "protocol": _protocol(args),
    }


def _validate_confirmation_roots(args: argparse.Namespace, output_root: Path) -> None:
    output_root = Path(output_root).resolve()
    _validate_roots(args, output_root)
    screen_root = (Path(args.data_root) / "fusion/experiments/f5-screen-v1").resolve()
    if (output_root == screen_root or screen_root in output_root.parents
            or output_root in screen_root.parents):
        raise ValueError("confirmation output must not overlap the F5 screen artifacts")


def execute_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    specs = confirmation_specs()
    output_root = Path(args.output_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    _validate_confirmation_roots(args, output_root)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")

    baseline_summaries, baseline_identity = _load_baseline_reference(baseline_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "matrix-status.json"
    aggregate_path = output_root / "aggregate-metrics.json"

    started_at = time.time()
    completed: list[str] = []
    failed: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    fold_inputs: dict[int, tuple[Any, Any, FeatureNormalizer, dict[str, Any], dict[str, str]]] = {}
    _write_matrix_status(status_path, state="running", specs=specs, completed=completed,
                         failed=failed, current=None, started_at=started_at)
    _atomic_json(aggregate_path, _aggregate(
        specs, completed, failed, summaries, state="running",
        baseline_summaries=baseline_summaries,
        baseline_identity=baseline_identity, started_at=started_at))

    for spec in specs:
        run_dir = output_root / spec.run_id
        config_path = run_dir / "config.json"
        status_file = run_dir / "training-status.json"
        checkpoint_dir = run_dir / "checkpoints"
        run_dir.mkdir(parents=True, exist_ok=True)
        preserve_status = False
        identity = None
        try:
            if spec.fold not in fold_inputs:
                fold_inputs[spec.fold] = _prepare_inputs(args, spec, baseline_root)
            train_data, validation_data, _normalizer, _fold_paths, cache_identity = fold_inputs[spec.fold]
            # The F2 source is seed-specific. Re-resolve it for each spec even
            # though datasets and normalizers are correctly cached per fold.
            paths = artifact_paths(args.data_root, baseline_root, spec)
            torch.manual_seed(spec.seed)
            model = build_baseline(spec.model)
            initialization = initialize_from_fold_f2(
                model, paths["f2_checkpoint"], fold=spec.fold, seed=spec.seed,
                split_manifest=paths["split_manifest"],
                normalization_path=paths["normalization"], limit=None)
            calibration_ranges = _ranges(train_data)
            config = _run_config(args, spec, paths, cache_identity, model,
                                 initialization, calibration_ranges)
            identity = config["identity"]

            if reusable_completed_run(run_dir, identity):
                previous = _read_json(status_file)
                summary = previous.get("screen_summary") or previous.get("confirmation_summary") or {
                    "run_id": spec.run_id, "model": spec.model, "fold": spec.fold,
                    "seed": spec.seed, "identity": identity,
                    "best_epoch": previous.get("best_epoch"),
                    "elapsed_seconds": previous.get("elapsed_seconds"),
                    "best_metrics": previous.get("best_metrics") or {},
                    "diagnostics": previous.get("diagnostics", {}),
                }
                summary["model"] = spec.model
                summary["fold"] = spec.fold
                summary["seed"] = spec.seed
                summaries[spec.run_id] = summary
                completed.append(spec.run_id)
            else:
                if config_path.exists():
                    previous_config = _read_json(config_path)
                    if previous_config.get("identity") != identity:
                        preserve_status = True
                        raise ValueError(f"existing run identity mismatch for {spec.run_id}; refusing overwrite")

                previous_status = {}
                if status_file.exists():
                    try:
                        previous_status = _read_json(status_file)
                    except (OSError, ValueError):
                        previous_status = {}
                if _prior_failure_blocks_retry(previous_status, args.retry_failed):
                    failed.append({
                        "run_id": spec.run_id,
                        "error": "previous attempt failed; pass --retry-failed to retry/resume",
                        "identity": identity,
                        "reused_failure": True,
                    })
                else:
                    resume_dir = None
                    checkpoint_meta_path = checkpoint_dir / "checkpoint.json"
                    if checkpoint_meta_path.is_file():
                        metadata = _read_json(checkpoint_meta_path)
                        if metadata.get("identity") != identity:
                            preserve_status = True
                            raise ValueError(f"checkpoint identity mismatch for {spec.run_id}")
                        if not (checkpoint_dir / "last.pt").is_file():
                            raise ValueError(f"resume metadata lacks last.pt for {spec.run_id}")
                        resume_dir = checkpoint_dir

                    _atomic_json(config_path, config)
                    _atomic_json(status_file, {
                        "state": "running", "run_id": spec.run_id,
                        "identity": identity, "attempt_started_unix": time.time(),
                        "locked_test_accessed": False,
                    })
                    _write_matrix_status(status_path, state="running", specs=specs,
                                         completed=completed, failed=failed,
                                         current=spec.run_id, started_at=started_at)
                    _atomic_json(aggregate_path, _aggregate(
                        specs, completed, failed, summaries, state="running",
                        baseline_summaries=baseline_summaries,
                        baseline_identity=baseline_identity, started_at=started_at))

                    with (run_dir / "stdout.log").open("a") as stdout, \
                            (run_dir / "stderr.log").open("a") as stderr:
                        print(f"RUN_BEGIN {spec.run_id} identity={identity} resume={resume_dir is not None}",
                              file=stdout, flush=True)
                        try:
                            run_started = time.monotonic()
                            result = train_fusion(
                                model, train_data, validation_data, run_dir,
                                config=config, epochs=args.epochs,
                                batch_size=args.batch_size, seed=spec.seed,
                                device=args.device, patience=args.patience,
                                min_epochs=args.min_epochs,
                                calibrators=DatasetCalibrationHeads(calibration_ranges),
                                resume=resume_dir, num_workers=args.num_workers,
                                status_interval_batches=args.status_interval_batches)
                            if result.get("state") != "completed":
                                raise RuntimeError(f"trainer returned non-completed state: {result.get('state')}")
                            checkpoint_meta = _read_json(checkpoint_dir / "checkpoint.json")
                            if checkpoint_meta.get("identity") != identity:
                                raise ValueError("completed checkpoint identity does not match planned run")
                            if not (checkpoint_dir / "best.pt").is_file():
                                raise FileNotFoundError("trainer completed without best.pt")
                            _assert_finite(result.get("best_metrics", {}), "best_metrics")
                            best_state = torch.load(checkpoint_dir / "best.pt",
                                                    map_location="cpu", weights_only=False)
                            model.load_state_dict(best_state["model"])
                            diagnostics = _candidate_diagnostics(
                                model, validation_data, args.device, args.batch_size)
                            f2_status = paths["f2_checkpoint"].parent.parent / "training-status.json"
                            source_elapsed = _read_json(f2_status).get("elapsed_seconds") if f2_status.is_file() else None
                            summary = {
                                "run_id": spec.run_id, "model": spec.model,
                                "fold": spec.fold, "seed": spec.seed,
                                "identity": identity,
                                "best_epoch": result.get("best_epoch"),
                                "elapsed_seconds": result.get("elapsed_seconds"),
                                "confirmation_wall_seconds": time.monotonic() - run_started,
                                "source_f2_elapsed_seconds": source_elapsed,
                                "best_metrics": result.get("best_metrics", {}),
                                "diagnostics": diagnostics,
                            }
                            final_status = _read_json(status_file)
                            final_status.update({
                                "run_id": spec.run_id, "identity": identity,
                                "locked_test_accessed": False,
                                "f2_initialization": initialization,
                                "confirmation_summary": summary,
                            })
                            _atomic_json(status_file, final_status)
                            summaries[spec.run_id] = summary
                            completed.append(spec.run_id)
                            print(f"RUN_COMPLETE {spec.run_id}", file=stdout, flush=True)
                        except Exception:
                            error = traceback.format_exc()
                            print(error, file=stderr, flush=True)
                            failed.append({"run_id": spec.run_id, "identity": identity,
                                           "error": error, "timestamp_unix": time.time(),
                                           "retry_requires_flag": True})
                            last_metrics = None
                            if status_file.exists():
                                try:
                                    last_metrics = _read_json(status_file).get("last_metrics")
                                except (OSError, ValueError):
                                    pass
                            _atomic_json(status_file, {
                                "state": "failed", "run_id": spec.run_id,
                                "identity": identity, "error": error,
                                "last_metrics": last_metrics,
                                "locked_test_accessed": False,
                                "f2_initialization": initialization,
                            })
        except Exception:
            error = traceback.format_exc()
            failed.append({"run_id": spec.run_id, "identity": identity,
                           "error": error, "timestamp_unix": time.time(),
                           "retry_requires_flag": True})
            if run_dir.exists() and not preserve_status:
                _atomic_json(status_file, {
                    "state": "failed", "run_id": spec.run_id,
                    "identity": identity, "error": error,
                    "locked_test_accessed": False,
                })

        _write_matrix_status(status_path, state="running", specs=specs,
                             completed=completed, failed=failed,
                             current=spec.run_id, started_at=started_at)
        _atomic_json(aggregate_path, _aggregate(
            specs, completed, failed, summaries, state="running",
            baseline_summaries=baseline_summaries,
            baseline_identity=baseline_identity, started_at=started_at))

    final_state = "completed" if not failed and len(completed) == len(specs) else "completed_with_failures"
    final = _aggregate(specs, completed, failed, summaries, state=final_state,
                       baseline_summaries=baseline_summaries,
                       baseline_identity=baseline_identity, started_at=started_at)
    _write_matrix_status(status_path, state=final_state, specs=specs,
                         completed=completed, failed=failed, current=None,
                         started_at=started_at)
    _atomic_json(aggregate_path, final)
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the fixed 30-run matrix; default is plan-only")
    parser.add_argument("--retry-failed", action="store_true",
                        help="retry failed runs with matching identity/checkpoint")
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
    args.output_root = args.output_root or Path(args.data_root) / "fusion/experiments/f5-confirmation-v1"
    try:
        _validate_confirmation_roots(args, args.output_root)
        result = execute_confirmation(args) if args.execute else plan_matrix(args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("state", "planned") in {"planned", "completed"} else 1
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
