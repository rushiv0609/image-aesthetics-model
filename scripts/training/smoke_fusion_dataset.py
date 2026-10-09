#!/usr/bin/env python3
"""Smoke every CV partition and optionally exercise spawned DataLoader workers."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from image_aesthetics.data.fusion_training_dataset import (
    DatasetBalancedBatchSampler,
    FusionTrainingDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-version", default="v5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--locked-only", action="store_true")
    args = parser.parse_args()
    split = args.data_root / "fusion" / "prepared" / args.split_version / "samples.jsonl"
    join_dirs = {dataset: args.data_root / dataset / "features" / "joined-ten-v1"
                 for dataset in ("aadb", "para", "flickr_aes", "tad66k")}
    for fold in ([] if args.locked_only else range(5)):
        train = FusionTrainingDataset(split, join_dirs, partition="train", validation_fold=fold)
        validation = FusionTrainingDataset(split, join_dirs, partition="validation", validation_fold=fold)
        sampler = DatasetBalancedBatchSampler(train, args.batch_size)
        batch = next(iter(DataLoader(train, batch_sampler=sampler, num_workers=args.workers)))
        assert tuple(batch["features"].shape) == (args.batch_size, 1544)
        assert batch["features"].isfinite().all()
        assert len(set(batch["dataset_id"])) == 1
        print(f"fold={fold} train={len(train)} validation={len(validation)} batch=ok")
    locked = FusionTrainingDataset(split, join_dirs, partition="locked_test", allow_locked_test=True)
    batch = next(iter(DataLoader(locked, batch_size=args.batch_size, num_workers=args.workers)))
    assert tuple(batch["features"].shape) == (args.batch_size, 1544)
    assert batch["features"].isfinite().all()
    print(f"locked_test={len(locked)} workers={args.workers} batch=ok")


if __name__ == "__main__":
    main()
