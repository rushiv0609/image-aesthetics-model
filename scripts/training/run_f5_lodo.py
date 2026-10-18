#!/usr/bin/env python3
"""Plan or execute leakage-safe leave-one-dataset-out F5/F4 evaluation.

The fixed outer matrix is four held-out datasets x two models x two seeds.
Fold 0 of the three source datasets selects checkpoints.  The held-out dataset
is absent from training, normalization, calibration-head fitting, early
stopping, and checkpoint selection; it is loaded only after training for final
development-set transfer evaluation.  F5 receives an internal source-only F2
initialization trained under the same exclusion policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.fusion_holdout import DatasetExclusionView
from image_aesthetics.models.fusion import DatasetCalibrationHeads, build_baseline
from image_aesthetics.training.fusion_trainer import _collect, train_fusion
from scripts.training.run_f5_screen import DATASETS, _atomic_json, _cache_hashes, _ranges

VERSION = "f5-lodo-v1"
SEEDS = (20260920, 20260921)
MODELS = ("F5-G", "F4")
SELECTION_FOLD = 0


@dataclass(frozen=True)
class LodoSpec:
    held_out_dataset: str
    model: str
    seed: int

    @property
    def run_id(self):
        return f"{self.model}-holdout-{self.held_out_dataset}-seed{self.seed}"


def lodo_specs(candidate: str = "F5-G"):
    models = (candidate, "F4")
    return [LodoSpec(dataset, model, seed) for dataset in DATASETS
            for model in models for seed in SEEDS]


def _paths(data_root: Path):
    return {"split": Path(data_root) / "fusion/prepared/v6/samples.jsonl",
            "joins": {name: Path(data_root) / name / "features/joined-ten-v1"
                      for name in DATASETS}}


def _protocol(args):
    return {"selection_fold": SELECTION_FOLD, "epochs": args.epochs,
            "patience": args.patience, "min_epochs": args.min_epochs,
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "device": args.device, "ranking_weight": 0.1,
            "held_out_used_for_checkpoint_selection": False}


def plan(args):
    specs = lodo_specs(args.candidate)
    return {"state": "planned", "experiment_version": VERSION, "total": len(specs),
            "candidate": args.candidate, "held_out_datasets": list(DATASETS),
            "models": [args.candidate, "F4"], "seeds": list(SEEDS),
            "selection_fold": SELECTION_FOLD, "locked_test_accessed": False,
            "runs": [{**asdict(spec), "run_id": spec.run_id} for spec in specs],
            "protocol": _protocol(args)}


def _architecture(model, name):
    if hasattr(model, "architecture_config"):
        return model.architecture_config()
    return {"model_id": name, "parameter_count": sum(p.numel() for p in model.parameters())}


def _build_candidate(name):
    if name == "F5-G-R15":
        return build_baseline("F5-G", residual_bound=0.15)
    if name == "F5-G-NoGate":
        return build_baseline("F5-G-NoGate", residual_bound=0.25)
    if name == "F5-G-Pair05":
        return build_baseline("F5-G", residual_bound=0.25)
    return build_baseline(name)


def _make_views(args, held_out, normalizer_path):
    paths = _paths(args.data_root)
    # Do not even open the held-out dataset's joined-feature index while
    # fitting source views, normalizers, calibrators, or checkpoints.
    source_joins = {name: path for name, path in paths["joins"].items()
                    if name != held_out}
    source_train = DatasetExclusionView(paths["split"], source_joins, partition="train",
                                        validation_fold=SELECTION_FOLD,
                                        excluded_dataset=held_out)
    source_validation = DatasetExclusionView(paths["split"], source_joins, partition="validation",
                                             validation_fold=SELECTION_FOLD,
                                             excluded_dataset=held_out)
    normalizer = source_train.fit_normalizer(normalizer_path)
    return (source_train.normalized(normalizer), source_validation.normalized(normalizer),
            normalizer, {"split": paths["split"], "joins": source_joins,
                         "held_out_join": paths["joins"][held_out]})


def _make_holdout_view(args, held_out, normalizer, paths):
    """Open held-out development features only after source checkpoint freeze."""
    held_parts = [DatasetExclusionView(
        paths["split"], {held_out: paths["held_out_join"]}, partition="validation",
        validation_fold=fold, only_dataset=held_out).normalized(normalizer)
        for fold in range(5)]
    return ConcatDataset(held_parts)


def _ranges_from(data):
    return _ranges(data)


def _identity(spec, paths, normalizer, args, architecture):
    payload = {"experiment_version": VERSION, "spec": asdict(spec),
               "split_sha256": hashlib.sha256(paths["split"].read_bytes()).hexdigest(),
               "cache_identity": _cache_hashes(paths["joins"]),
               "normalization_stats_hash": normalizer.stats.stats_hash,
               "normalization_identity": dict(normalizer.stats.identity),
               "architecture": architecture, "protocol": _protocol(args),
               "held_out_dataset": spec.held_out_dataset,
               "held_out_absent_from_training_and_selection": True,
               "locked_test_accessed": False}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                      default=str).encode()).hexdigest()
    return digest, payload


def _assert_exclusion(dataset, held_out):
    visible = {sample[0] for sample in dataset.samples}
    if held_out in visible or visible != set(DATASETS) - {held_out}:
        raise ValueError(f"source view violates holdout policy: {held_out}, visible={visible}")


def execute(args):
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    output = Path(args.output_root).resolve(); data_root = Path(args.data_root).resolve()
    repo_root = ROOT.resolve()
    if output == repo_root or repo_root in output.parents:
        raise ValueError("LODO output must stay outside the Git repository")
    if output == data_root or output in data_root.parents:
        raise ValueError("LODO output must not overlap the dataset root")
    output.mkdir(parents=True, exist_ok=True)
    completed, failed, summaries = [], [], {}
    specs = lodo_specs(args.candidate)
    for spec in specs:
        run_dir = output / spec.run_id; run_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_data, validation_data, normalizer, paths = _make_views(
                args, spec.held_out_dataset, run_dir / "normalization.json")
            _assert_exclusion(train_data.dataset, spec.held_out_dataset)
            _assert_exclusion(validation_data.dataset, spec.held_out_dataset)
            torch.manual_seed(spec.seed)
            model = _build_candidate(spec.model)
            initialization = None
            if spec.model.startswith("F5-"):
                # Internal source-only F2 prerequisite.  It is never evaluated
                # as an outer candidate and never sees held-out rows.
                f2_dir = run_dir / "source-only-f2"
                f2 = build_baseline("F2")
                f2_config = {"identity": hashlib.sha256(
                    json.dumps({"run_id": spec.run_id, "stage": "source-only-f2",
                                "normalization": normalizer.stats.stats_hash,
                                "protocol": _protocol(args)}, sort_keys=True).encode()).hexdigest(),
                    "held_out_dataset": spec.held_out_dataset,
                    "held_out_absent": True, "locked_test_accessed": False}
                f2_status_path = f2_dir / "training-status.json"
                f2_checkpoint_meta = f2_dir / "checkpoints/checkpoint.json"
                f2_resume = None
                if f2_status_path.is_file() and json.loads(f2_status_path.read_text()).get("state") == "completed":
                    f2_result = json.loads(f2_status_path.read_text())
                    metadata = json.loads(f2_checkpoint_meta.read_text())
                    if (metadata.get("identity") != f2_config["identity"]
                            or not (f2_dir / "checkpoints/best.pt").is_file()):
                        raise ValueError(f"completed source-only F2 identity mismatch: {spec.run_id}")
                else:
                    if (f2_checkpoint_meta.is_file()
                            and (f2_dir / "checkpoints/last.pt").is_file()):
                        metadata = json.loads(f2_checkpoint_meta.read_text())
                        if metadata.get("identity") != f2_config["identity"]:
                            raise ValueError(f"source-only F2 identity mismatch: {spec.run_id}")
                        f2_resume = f2_dir / "checkpoints"
                    f2_result = train_fusion(
                        f2, train_data, validation_data, f2_dir, config=f2_config,
                        epochs=args.epochs, batch_size=args.batch_size, seed=spec.seed,
                        device=args.device, patience=args.patience, min_epochs=args.min_epochs,
                        calibrators=DatasetCalibrationHeads(_ranges_from(train_data)),
                        num_workers=args.num_workers, resume=f2_resume,
                        status_interval_batches=args.status_interval_batches)
                    if f2_result.get("state") != "completed":
                        raise RuntimeError("source-only F2 training did not complete")
                state = torch.load(f2_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
                model.initialize_from_f2(state["model"])
                initialization = {"kind": "source-only-lodo-F2", "identity": f2_config["identity"],
                                  "best_epoch": f2_result["best_epoch"]}
            architecture = _architecture(model, spec.model)
            identity, payload = _identity(spec, paths, normalizer, args, architecture)
            evaluation_path = run_dir / "held-out-evaluation.json"
            config_path = run_dir / "config.json"
            status_path = run_dir / "training-status.json"
            if evaluation_path.is_file() and config_path.is_file() and status_path.is_file():
                prior_config = json.loads(config_path.read_text())
                prior_status = json.loads(status_path.read_text())
                prior_evaluation = json.loads(evaluation_path.read_text())
                if (prior_config.get("identity") == identity
                        and prior_status.get("state") == "completed"
                        and prior_evaluation.get("identity") == identity):
                    summaries[spec.run_id] = prior_evaluation
                    completed.append(spec.run_id)
                    continue
                raise ValueError(f"existing LODO artifacts do not match identity: {spec.run_id}")
            config = {"identity": identity, "identity_payload": payload,
                      "run_id": spec.run_id, "model": spec.model,
                      "seed": spec.seed, "f2_initialization": initialization,
                      "locked_test_accessed": False}
            _atomic_json(run_dir / "config.json", config)
            ranking_gap = 0.05 if spec.model == "F5-G-Pair05" else 0.0
            resume = None
            checkpoint_meta = run_dir / "checkpoints/checkpoint.json"
            if checkpoint_meta.is_file() and (run_dir / "checkpoints/last.pt").is_file():
                metadata = json.loads(checkpoint_meta.read_text())
                if metadata.get("identity") != identity:
                    raise ValueError(f"LODO checkpoint identity mismatch: {spec.run_id}")
                resume = run_dir / "checkpoints"
            result = train_fusion(
                model, train_data, validation_data, run_dir, config=config,
                epochs=args.epochs, batch_size=args.batch_size, seed=spec.seed,
                device=args.device, patience=args.patience, min_epochs=args.min_epochs,
                calibrators=DatasetCalibrationHeads(_ranges_from(train_data)),
                ranking_min_target_gap=ranking_gap,
                resume=resume,
                num_workers=args.num_workers,
                status_interval_batches=args.status_interval_batches)
            if result.get("state") != "completed":
                raise RuntimeError("source candidate training did not complete")
            state = torch.load(run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"]); model.to(args.device)
            holdout_data = _make_holdout_view(args, spec.held_out_dataset, normalizer, paths)
            report = _collect(model, DataLoader(holdout_data, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers),
                              torch.device(args.device), calibrators=None)
            summary = {"run_id": spec.run_id, **asdict(spec), "identity": identity,
                       "selection_best_epoch": result["best_epoch"],
                       "held_out_metrics": report, "locked_test_accessed": False}
            _atomic_json(run_dir / "held-out-evaluation.json", summary)
            summaries[spec.run_id] = summary; completed.append(spec.run_id)
        except Exception as exc:
            failed.append({"run_id": spec.run_id, "error": repr(exc)})
        _atomic_json(output / "matrix-status.json", {
            "state": "running", "total": len(specs), "completed": completed,
            "failed": failed, "pending": [s.run_id for s in specs
                                           if s.run_id not in completed and
                                           s.run_id not in {f['run_id'] for f in failed}],
            "locked_test_accessed": False})
    state = "completed" if len(completed) == len(specs) and not failed else "completed_with_failures"
    aggregate = {"state": state, "experiment_version": VERSION, "total": len(specs),
                 "completed": completed, "failed": failed, "summaries": summaries,
                 "locked_test_accessed": False}
    _atomic_json(output / "aggregate-metrics.json", aggregate)
    _atomic_json(output / "matrix-status.json", {**aggregate, "summaries": None})
    return aggregate


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate", choices=("F5-G", "F5-X", "F5-G-R15",
                                                  "F5-G-NoGate", "F5-G-Pair05"),
                        default="F5-G")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--status-interval-batches", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-epochs", type=int, default=8)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    args.output_root = args.output_root or (Path(args.data_root) / "fusion/experiments/f5-lodo-v1")
    value = execute(args) if args.execute else plan(args)
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0 if value["state"] in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
