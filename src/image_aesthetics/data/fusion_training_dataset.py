"""Leakage-safe fusion training views over joined-ten feature indexes."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Mapping

from torch.utils.data import Dataset, Sampler

from image_aesthetics.data.joined_ten_dataset import JoinedTenDataset


class FusionTrainingDataset(Dataset):
    """Join a frozen fusion split to cached features without exposing test rows."""

    def __init__(self, split_manifest: Path, join_dirs: Mapping[str, Path], *,
                 partition: str, validation_fold: int | None = None,
                 allow_locked_test: bool = False, max_open_shards: int = 8):
        if partition not in {"train", "validation", "locked_test"}:
            raise ValueError(f"unknown partition: {partition}")
        if partition in {"train", "validation"} and validation_fold not in range(5):
            raise ValueError("development partitions require validation_fold in [0, 4]")
        if partition == "locked_test" and not allow_locked_test:
            raise PermissionError("locked test requires explicit opt-in")
        rows = [json.loads(line) for line in Path(split_manifest).read_text().splitlines() if line]
        if partition == "locked_test":
            rows = [row for row in rows if row["locked_test"]]
        elif partition == "validation":
            rows = [row for row in rows if not row["locked_test"] and row["development_fold"] == validation_fold]
        else:
            rows = [row for row in rows if not row["locked_test"] and row["development_fold"] != validation_fold]
        self.datasets = {name: JoinedTenDataset(path, max_open_shards=max_open_shards)
                         for name, path in join_dirs.items()}
        indexes = {name: {str(row["image_id"]): index for index, row in enumerate(dataset.rows)}
                   for name, dataset in self.datasets.items()}
        self.samples = []
        for row in rows:
            dataset_id = str(row["dataset_id"])
            feature_id = str(row.get("feature_id") or row["image_id"])
            if dataset_id not in indexes or feature_id not in indexes[dataset_id]:
                raise ValueError(f"missing joined-ten feature for {dataset_id}:{feature_id}")
            low, high = map(float, row["native_score_range"])
            if high <= low:
                raise ValueError(f"invalid target range for {feature_id}")
            self.samples.append((dataset_id, indexes[dataset_id][feature_id], row,
                                 (float(row["native_score"]) - low) / (high - low)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        dataset_id, feature_index, target_row, target = self.samples[index]
        item = self.datasets[dataset_id][feature_index]
        return {
            "features": item["features"],
            "target": target,
            "native_target": float(target_row["native_score"]),
            "dataset_id": dataset_id,
            "image_id": item["image_id"],
            "group_id": str(target_row["group_id"]),
            "development_fold": target_row["development_fold"],
        }


class DatasetBalancedBatchSampler(Sampler[list[int]]):
    """Yield deterministic, dataset-homogeneous batches with equal dataset turns."""

    def __init__(self, dataset: FusionTrainingDataset, batch_size: int, *, seed: int = 20260920,
                 drop_last: bool = False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (dataset_id, *_rest) in enumerate(dataset.samples):
            grouped[dataset_id].append(index)
        if not grouped:
            raise ValueError("cannot sample an empty dataset")
        self.grouped = dict(sorted(grouped.items()))
        largest = max(len(values) for values in self.grouped.values())
        self.batches_per_dataset = (largest // batch_size if drop_last else math.ceil(largest / batch_size))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_dataset * len(self.grouped)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pools = {}
        positions = {}
        for name, values in self.grouped.items():
            pools[name] = values.copy()
            rng.shuffle(pools[name])
            positions[name] = 0
        for _ in range(self.batches_per_dataset):
            for name in self.grouped:
                batch = []
                while len(batch) < self.batch_size:
                    if positions[name] == len(pools[name]):
                        positions[name] = 0
                        rng.shuffle(pools[name])
                    remaining = self.batch_size - len(batch)
                    take = min(remaining, len(pools[name]) - positions[name])
                    batch.extend(pools[name][positions[name]:positions[name] + take])
                    positions[name] += take
                if batch:
                    yield batch
