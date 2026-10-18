#!/usr/bin/env python3
"""Export one prediction per development image from fold-selected checkpoints.

Predictions are ensembled across the requested seeds within each image's own
validation fold. The output is suitable for shared monotonic calibration and
never constructs the locked-test partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import build_baseline
from scripts.training.run_f5_screen import DATASETS


def _build(name: str, config: dict):
    architecture = config.get("model_config") or config.get("identity_payload", {}).get("architecture", {})
    bound = float(architecture.get("residual_bound", 0.25))
    if name == "F5-G-R15":
        return build_baseline("F5-G", residual_bound=bound)
    if name == "F5-G-NoGate":
        return build_baseline("F5-G-NoGate", residual_bound=bound)
    if name == "F5-G-Pair05":
        return build_baseline("F5-G", residual_bound=bound)
    return build_baseline(name, **({"residual_bound": bound} if name.startswith("F5-") else {}))


def _validated_run(root: Path, candidate: str, fold: int, seed: int):
    run_id = f"{candidate}-fold{fold}-seed{seed}"
    run = root / run_id
    config = json.loads((run / "config.json").read_text())
    status = json.loads((run / "training-status.json").read_text())
    metadata = json.loads((run / "checkpoints/checkpoint.json").read_text())
    identity = config.get("identity")
    if (config.get("run_id") != run_id or int(config.get("fold", -1)) != fold
            or config.get("model") != candidate
            or int(config.get("seed", -1)) != seed or status.get("state") != "completed"
            or status.get("identity") != identity or metadata.get("identity") != identity
            or config.get("locked_test_accessed") is not False
            or status.get("locked_test_accessed") is not False):
        raise ValueError(f"incomplete or inconsistent OOF source: {run_id}")
    checkpoint = run / "checkpoints/best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return run_id, run, config, identity, checkpoint


def export(args):
    joins = {name: Path(args.data_root) / name / "features/joined-ten-v1" for name in DATASETS}
    split = Path(args.data_root) / "fusion/prepared/v6/samples.jsonl"
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a non-empty unique comma-separated list")
    device = torch.device(args.device)
    rows = []
    source_identities = {}
    for fold in range(5):
        sources = [_validated_run(args.experiment_root, args.candidate, fold, seed)
                   for seed in seeds]
        config = sources[0][2]
        normalization_path = Path(config["normalization_path"])
        base = FusionTrainingDataset(split, joins, partition="validation", validation_fold=fold)
        normalizer = FeatureNormalizer.from_file(
            normalization_path, expected_identity=FusionTrainingDataset(
                split, joins, partition="train", validation_fold=fold).normalization_identity())
        data = base.normalized(normalizer)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        predictions = []
        for run_id, _run, run_config, identity, checkpoint in sources:
            model = _build(args.candidate, run_config)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"]); model.to(device); model.eval()
            batches = []
            with torch.no_grad():
                for batch in loader:
                    batches.append(model(batch["features"].to(device=device,
                                                               dtype=torch.float32)).cpu())
            predictions.append(torch.cat(batches))
            source_identities[run_id] = {"identity": identity,
                                         "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        ensemble = torch.stack(predictions).mean(0).tolist()
        for sample, prediction in zip(base.samples, ensemble):
            dataset_id, _feature_index, target_row, target = sample
            if target_row.get("locked_test"):
                raise ValueError("locked-test row reached OOF export")
            rows.append({"dataset_id": dataset_id,
                         "image_id": str(target_row["image_id"]),
                         "group_id": str(target_row["group_id"]),
                         "development_fold": fold, "source_validation_fold": fold,
                         "prediction": float(prediction), "target": float(target),
                         "locked_test": False, "candidate": args.candidate,
                         "source_seeds": list(seeds)})
    rows.sort(key=lambda row: (row["dataset_id"], row["image_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    provenance = {"schema_version": "f5-oof-predictions-v1", "candidate": args.candidate,
                  "seeds": list(seeds), "n_rows": len(rows),
                  "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
                  "source_runs": source_identities, "locked_test_accessed": False,
                  "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(args.output.suffix + ".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--seeds", default="20260920,20260921,20260922")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
