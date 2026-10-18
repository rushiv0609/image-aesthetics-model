#!/usr/bin/env python3
"""Plan or run the frozen, fixed-epoch F5 fit on all development rows.

Planning is the default and reads only the frozen policy. Execution never
constructs the locked-test dataset and never selects checkpoints from labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from image_aesthetics.data.fusion_training_dataset import (
    DatasetBalancedBatchSampler, FusionTrainingDataset,
)
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import DatasetCalibrationHeads, build_baseline
from image_aesthetics.models.fusion.losses import calibrated_huber_loss, pairwise_ranking_loss
from image_aesthetics.training.f5_finalization import (
    assert_external_path, atomic_json, load_frozen_policy, sha256_file,
)
from scripts.training.run_f5_screen import _cache_hashes, _ranges


def plan(policy_path: Path, output_root: Path) -> dict:
    policy, digest = load_frozen_policy(policy_path)
    return {"state": "planned", "policy_sha256": digest,
            "candidate": policy["candidate"], "epochs": policy["epochs"],
            "seeds": policy["seeds"], "output_root": str(output_root),
            "device": policy["training"]["device"],
            "partition": "full-development", "locked_test_accessed": False,
            "fit_on_execute": True}


def _fit_stage(model, dataset, calibration_heads, *, epochs: int, seed: int,
               batch_size: int, device: torch.device, ranking_weight: float,
               ranking_min_target_gap: float, num_workers: int, output: Path) -> dict:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    sampler = DatasetBalancedBatchSampler(dataset, batch_size, seed=seed)
    loader_options = {"num_workers": num_workers}
    if num_workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_options)
    model.to(device); calibration_heads.to(device)
    params = list(model.parameters()) + list(calibration_heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    total_steps = max(1, epochs * len(loader)); warmup = max(1, int(.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup))
    steps = 0; epoch_losses = []; started = time.time()
    for epoch in range(epochs):
        sampler.set_epoch(epoch); model.train(); calibration_heads.train()
        loss_sum = 0.0; batches = 0
        for batch in loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            prediction = model(features)
            if prediction.ndim > 1:
                prediction = prediction.mean(-1)
            loss, _ = calibrated_huber_loss(
                prediction, batch["native_target"].to(device=device, dtype=torch.float32),
                list(batch["dataset_id"]), calibration_heads)
            if ranking_weight:
                pair_groups = batch.get("pairing_group")
                if pair_groups is not None and len(set(pair_groups)) == 1:
                    pair_groups = None
                loss = loss + ranking_weight * pairwise_ranking_loss(
                    prediction, batch["target"].to(device=device, dtype=torch.float32),
                    groups=pair_groups, min_target_gap=ranking_min_target_gap)
            if hasattr(model, "regularization_loss"):
                loss = loss + model.regularization_loss()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); steps += 1
            if steps > warmup:
                scheduler.step()
            loss_sum += float(loss.detach().cpu()); batches += 1
        epoch_losses.append(loss_sum / max(1, batches))
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "final.pt"
    temporary = state_path.with_suffix(".pt.tmp")
    torch.save({"model": model.cpu().state_dict(),
                "calibrators": calibration_heads.cpu().state_dict(),
                "seed": seed, "epochs": epochs, "steps": steps}, temporary)
    temporary.replace(state_path)
    return {"checkpoint": str(state_path), "checkpoint_sha256": sha256_file(state_path),
            "epochs": epochs, "steps": steps, "epoch_losses": epoch_losses,
            "elapsed_seconds": time.time() - started}


def execute(args) -> dict:
    policy, policy_digest = load_frozen_policy(args.policy)
    data_root = Path(args.data_root).resolve()
    output = assert_external_path(args.output_root, repo_root=ROOT, data_root=data_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"final-fit output must be a new or empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    split = data_root / "fusion/prepared/v6/samples.jsonl"
    join_dirs = {name: data_root / name / "features/joined-ten-v1"
                 for name in policy["training"]["datasets"]}
    if set(join_dirs) != set(policy["feature_hashes"]):
        raise ValueError("policy feature_hashes must exactly cover the training datasets")
    if not split.is_file() or sha256_file(split) != policy["split_sha256"]:
        raise ValueError("development split manifest differs from frozen policy")
    actual_features = _cache_hashes(join_dirs)
    if actual_features != policy["feature_hashes"]:
        raise ValueError("joined feature identities differ from frozen policy")
    if args.device and args.device != policy["training"]["device"]:
        raise ValueError("requested device differs from the frozen policy")
    device = torch.device(policy["training"]["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")

    raw_development = FusionTrainingDataset(
        split, join_dirs, partition="development", dataset_ids=policy["training"]["datasets"])
    if not raw_development.samples:
        raise ValueError("frozen policy produced an empty full-development fit")
    if any(row.get("locked_test") is not False for _ds, _i, row, _y in raw_development.samples):
        raise ValueError("locked-test row entered full-development fitting")
    normalizer_path = output / "normalization.json"
    normalizer = raw_development.fit_normalizer(normalizer_path)
    dataset = raw_development.normalized(normalizer)
    ranges = _ranges(raw_development)
    result = {
        "schema_version": "f5-full-development-fit-v1", "state": "running",
        "policy_sha256": policy_digest, "candidate": policy["candidate"],
        "split_sha256": policy["split_sha256"],
        "feature_hashes": actual_features,
        "normalization_sha256": sha256_file(normalizer_path),
        "normalization_stats_hash": normalizer.stats.stats_hash,
        "development_rows": len(raw_development), "locked_test_accessed": False,
        "seeds": {},
    }
    atomic_json(output / "fit-status.json", result)
    for seed in policy["seeds"]:
        seed_dir = output / f"seed-{seed}"
        f2 = build_baseline("F2")
        f2_fit = _fit_stage(
            f2, dataset, DatasetCalibrationHeads(ranges), epochs=policy["epochs"]["f2"],
            seed=seed, batch_size=int(policy["training"]["batch_size"]), device=device,
            ranking_weight=float(policy["training"]["ranking_weight"]),
            ranking_min_target_gap=float(policy["training"]["ranking_min_target_gap"]),
            num_workers=args.num_workers, output=seed_dir / "f2")
        candidate = build_baseline(policy["candidate"], **policy["architecture"].get("kwargs", {}))
        actual_architecture = candidate.architecture_config()
        for key, value in policy["architecture"].items():
            if key != "kwargs" and actual_architecture.get(key) != value:
                raise ValueError(f"instantiated architecture differs from policy field {key}")
        for key, value in policy["architecture"].get("kwargs", {}).items():
            if actual_architecture.get(key) != value:
                raise ValueError(f"instantiated architecture differs from frozen kwarg {key}")
        state = torch.load(f2_fit["checkpoint"], map_location="cpu", weights_only=False)
        candidate.initialize_from_f2(state["model"])
        candidate_fit = _fit_stage(
            candidate, dataset, DatasetCalibrationHeads(ranges),
            epochs=policy["epochs"]["candidate"], seed=seed,
            batch_size=int(policy["training"]["batch_size"]), device=device,
            ranking_weight=float(policy["training"]["ranking_weight"]),
            ranking_min_target_gap=float(policy["training"]["ranking_min_target_gap"]),
            num_workers=args.num_workers, output=seed_dir / "candidate")
        result["seeds"][str(seed)] = {"f2": f2_fit, "candidate": candidate_fit}
        result["state"] = "running"
        atomic_json(output / "fit-status.json", result)
    result["state"] = "completed"
    result["checkpoint_policy"] = dict(policy["checkpoint_policy"])
    result["completed_at_unix"] = time.time()
    atomic_json(output / "fit-status.json", result)
    result["fit_manifest_sha256"] = sha256_file(output / "fit-status.json")
    atomic_json(output / "fit-manifest.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="fit fixed epochs on all development data")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default=None,
                        help="optional assertion; training uses the frozen policy device")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    assert_external_path(args.policy, repo_root=ROOT, data_root=args.data_root)
    value = execute(args) if args.execute else plan(args.policy, args.output_root)
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0 if value["state"] in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
