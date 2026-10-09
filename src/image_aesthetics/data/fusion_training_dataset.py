"""Leakage-safe fusion training views over joined-ten feature indexes."""

from __future__ import annotations

import json
import hashlib
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from torch.utils.data import Dataset, Sampler

from image_aesthetics.data.joined_ten_dataset import JoinedTenDataset
from image_aesthetics.data.normalization import FeatureNormalizer


class FusionTrainingDataset(Dataset):
    """Join a frozen fusion split to cached features without exposing test rows."""

    def __init__(self, split_manifest: Path, join_dirs: Mapping[str, Path], *,
                 partition: str, validation_fold: int | None = None,
                 allow_locked_test: bool = False, max_open_shards: int = 8,
                 normalizer: FeatureNormalizer | None = None):
        if partition not in {"train", "validation", "locked_test"}:
            raise ValueError(f"unknown partition: {partition}")
        if partition in {"train", "validation"} and validation_fold not in range(5):
            raise ValueError("development partitions require validation_fold in [0, 4]")
        if partition == "locked_test" and not allow_locked_test:
            raise PermissionError("locked test requires explicit opt-in")
        self.partition = partition
        self.validation_fold = validation_fold
        self.split_manifest = Path(split_manifest)
        self.join_dirs = dict(join_dirs)
        self.normalizer = normalizer
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

    def normalization_identity(self) -> dict[str, Any]:
        """Identity used to prevent reusing stats fitted for another fold."""
        digest = hashlib.sha256(self.split_manifest.read_bytes()).hexdigest()
        join_hashes = {}
        for name, path in sorted(self.join_dirs.items()):
            join_digest = hashlib.sha256()
            for filename in ("join-audit.json", "join.jsonl"):
                candidate = Path(path) / filename
                if candidate.is_file():
                    join_digest.update(candidate.read_bytes())
            join_hashes[name] = join_digest.hexdigest()
        return {
            "schema_version": "fusion-training-view-v1",
            "split_manifest_sha256": digest,
            "partition": self.partition,
            "validation_fold": self.validation_fold,
            "dataset_ids": sorted(self.datasets),
            "join_hashes": join_hashes,
            "feature_dimension": 1544,
        }

    def fit_normalizer(self, stats_path: Path | None = None) -> FeatureNormalizer:
        """Fit only this view's development-train rows, streaming features."""
        if self.partition != "train":
            raise ValueError("feature normalization must be fitted on a train partition")
        identity = self.normalization_identity()
        return FeatureNormalizer.fit((self[index]["features"] for index in range(len(self))),
                                     identity=identity, stats_path=stats_path)

    def normalized(self, normalizer: FeatureNormalizer) -> "NormalizedFusionTrainingDataset":
        return NormalizedFusionTrainingDataset(self, normalizer)

    def ranking_pairs(self, *, seed: int = 20260920, max_pairs: int | None = 1_000_000,
                      allow_locked_test: bool = False) -> list[dict[str, Any]]:
        """Build compatible pairs from the rows visible in this partition."""
        return build_ranking_pairs((sample[2] for sample in self.samples), seed=seed,
                                   max_pairs=max_pairs, allow_locked_test=allow_locked_test)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        dataset_id, feature_index, target_row, target = self.samples[index]
        item = self.datasets[dataset_id][feature_index]
        features = item["features"] if self.normalizer is None else self.normalizer.transform(item["features"])
        result = {
            "features": features,
            "target": target,
            "native_target": float(target_row["native_score"]),
            "dataset_id": dataset_id,
            "image_id": item["image_id"],
            "group_id": str(target_row["group_id"]),
            "pairing_group": (f"tad66k:{(_themes(target_row) or ('__theme_less__',))[0]}"
                              if dataset_id == "tad66k" else dataset_id),
            # PyTorch's default collator cannot batch None values.  Locked-test
            # rows have no development fold, represented as the explicit -1
            # sentinel at the loader boundary.
            "development_fold": (-1 if target_row["development_fold"] is None
                                 else int(target_row["development_fold"])),
        }
        if self.normalizer is not None:
            result["normalization_stats_hash"] = self.normalizer.stats.stats_hash
        return result


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


class NormalizedFusionTrainingDataset(Dataset):
    """Lazy wrapper that normalizes each joined feature row at access time."""

    def __init__(self, dataset: Dataset, normalizer: FeatureNormalizer):
        self.dataset = dataset
        self.normalizer = (normalizer if isinstance(normalizer, FeatureNormalizer)
                           else FeatureNormalizer(normalizer))
        if hasattr(dataset, "samples"):
            self.samples = dataset.samples

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        result = dict(self.dataset[index])
        result["features"] = self.normalizer.transform(result["features"])
        result["normalization_stats_hash"] = self.normalizer.stats.stats_hash
        return result


def _themes(row: Mapping[str, Any]) -> tuple[str, ...]:
    values = row.get("themes") or row.get("theme") or row.get("theme_id")
    if values is None:
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            values = metadata.get("themes") or metadata.get("theme") or metadata.get("theme_id")
    if isinstance(values, str):
        values = [values]
    if not values:
        return ()
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("record_id") or row.get("feature_id") or row.get("image_id"))


def build_ranking_pairs(rows: Iterable[Mapping[str, Any]], *, seed: int = 20260920,
                        max_pairs: int | None = 1_000_000, allow_locked_test: bool = False) -> list[dict[str, Any]]:
    """Build deterministic same-dataset ranking pairs.

    TAD66K pairs additionally require a shared theme.  Other datasets remain in
    their native label space and are paired only within that dataset.  Locked
    rows are excluded unless explicitly requested.
    """
    usable = []
    for row in rows:
        if row.get("locked_test") and not allow_locked_test:
            continue
        dataset_id = str(row.get("dataset_id"))
        if not dataset_id or row.get("native_score") is None:
            continue
        themes = _themes(row)
        usable.append((dataset_id, _row_id(row), float(row["native_score"]), themes, row))
    groups: dict[tuple[str, str], list[tuple[str, str, float, tuple[str, ...], Mapping[str, Any]]]] = defaultdict(list)
    for item in usable:
        dataset_id, _, _, themes, _ = item
        if dataset_id == "tad66k":
            # Enforce theme compatibility when TAD provides it.  Theme-less
            # rows remain pairable only with other theme-less TAD rows.
            if themes:
                for theme in themes:
                    groups[(dataset_id, theme)].append(item)
            else:
                groups[(dataset_id, "__dataset__")].append(item)
        else:
            groups[(dataset_id, "__dataset__")].append(item)
    if max_pairs is not None and max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    group_items = sorted(groups.items())
    total_possible = sum(len(members) * (len(members) - 1) // 2 for _, members in group_items)
    bounded = max_pairs is not None and total_possible > max_pairs
    per_group_budget = (math.ceil(max_pairs / max(1, len(group_items))) if bounded else None)

    def add_pair(dataset_id, theme, left, right):
        if left[2] == right[2]:
            return
        preferred, other = (left, right) if left[2] > right[2] else (right, left)
        pair_key = (preferred[1], other[1])
        candidates[pair_key] = {
            "pair_id": hashlib.sha256(f"{dataset_id}:{theme}:{preferred[1]}:{other[1]}".encode()).hexdigest()[:24],
            "dataset_id": dataset_id,
            "theme": None if theme == "__dataset__" else theme,
            "preferred_id": preferred[1], "other_id": other[1],
            "positive_id": preferred[1], "negative_id": other[1],
            "left_id": preferred[1], "right_id": other[1],
            "preferred_score": preferred[2], "other_score": other[2],
            "margin": abs(preferred[2] - other[2]),
        }

    for (dataset_id, theme), members in group_items:
        members = sorted(members, key=lambda item: item[1])
        if len(members) < 2:
            continue
        if not bounded:
            for left_index in range(len(members)):
                for right_index in range(left_index + 1, len(members)):
                    add_pair(dataset_id, theme, members[left_index], members[right_index])
            continue
        # Cover the score range deterministically, then add reproducible random
        # pairs.  This is O(max_pairs), never O(number_of_images squared).
        ranked = sorted(members, key=lambda item: (item[2], item[1]))
        before = len(candidates)
        for index in range(len(ranked) - 1):
            add_pair(dataset_id, theme, ranked[index], ranked[index + 1])
            if len(candidates) - before >= per_group_budget:
                break
        rng = random.Random(f"{seed}:{dataset_id}:{theme}")
        attempts = 0
        while len(candidates) - before < per_group_budget and attempts < per_group_budget * 20:
            left_index, right_index = rng.sample(range(len(members)), 2)
            add_pair(dataset_id, theme, members[left_index], members[right_index])
            attempts += 1
    output = list(candidates.values())
    output.sort(key=lambda pair: (
        hashlib.sha256(f"{seed}:{pair['pair_id']}".encode()).hexdigest(),
        pair["dataset_id"], pair["preferred_id"], pair["other_id"],
    ))
    if max_pairs is not None:
        output = output[:max_pairs]
    output.sort(key=lambda pair: (pair["dataset_id"], pair["theme"] or "", pair["preferred_id"], pair["other_id"]))
    return output


generate_ranking_pairs = build_ranking_pairs
compatible_ranking_pairs = build_ranking_pairs
build_compatible_pairs = build_ranking_pairs


def _spaq_group(row: Mapping[str, Any]) -> str:
    explicit = row.get("technical_group_id") or row.get("group_id")
    if explicit:
        return str(explicit)
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    device = row.get("device_id") or row.get("phone_model") or metadata.get("device_id") or metadata.get("phone_model")
    scene = row.get("scene") or row.get("scene_id") or metadata.get("scene") or metadata.get("scene_id")
    if device or scene:
        return f"spaq:{device or 'unknown-device'}:{scene or 'unknown-scene'}"
    source_groups = row.get("source_groups")
    if isinstance(source_groups, str):
        source_groups = [source_groups]
    if source_groups:
        return str(sorted(str(value) for value in source_groups)[0])
    return f"spaq:image:{_row_id(row)}"


def prepare_spaq_technical_splits(rows: Iterable[Mapping[str, Any]], *, folds: int = 5,
                                  seed: int = 20260920, locked_fraction: float = 0.1) -> list[dict[str, Any]]:
    """Create deterministic group-disjoint SPAQ locked/dev/fold partitions."""
    if folds < 2 or not 0 <= locked_fraction < 1:
        raise ValueError("folds must be >=2 and locked_fraction must be in [0,1)")
    original = [dict(row) for row in rows]
    if any(str(row.get("dataset_id")) != "spaq" for row in original):
        raise ValueError("SPAQ technical split input must contain only dataset_id='spaq'")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original:
        row["group_id"] = _spaq_group(row)
        groups[row["group_id"]].append(row)
    locked_groups = {
        row["group_id"] for row in original
        if bool(row.get("locked_test") or row.get("upstream_locked_test") or
                str(row.get("split") or row.get("source_split") or "").lower() in {"test", "locked-test", "locked_test"})
    }
    target_locked = max(sum(len(groups[group]) for group in locked_groups), round(len(original) * locked_fraction))
    for group in sorted(groups, key=lambda value: (hashlib.sha256(f"locked:{seed}:{value}".encode()).hexdigest(), value)):
        if sum(len(groups[value]) for value in locked_groups) >= target_locked:
            break
        locked_groups.add(group)
    development_groups = [group for group in groups if group not in locked_groups]
    # Largest groups first makes the result less skewed; hash is the stable tie breaker.
    development_groups.sort(key=lambda value: (-len(groups[value]), hashlib.sha256(f"fold:{seed}:{value}".encode()).hexdigest(), value))
    fold_sizes = [0] * folds
    assignments: dict[str, int] = {}
    for group in development_groups:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        assignments[group] = fold
        fold_sizes[fold] += len(groups[group])
    result = []
    for row in sorted(original, key=lambda value: _row_id(value)):
        value = dict(row)
        value["task"] = "technical"
        value["label_task"] = "technical"
        value["training_role"] = "technical_auxiliary"
        value["eligible_for_fusion"] = False
        value["eligible_as_aesthetic_target"] = False
        value["locked_test"] = value["group_id"] in locked_groups
        value["development_fold"] = None if value["locked_test"] else assignments[value["group_id"]]
        low, high = value.get("native_score_range", [0.0, 100.0])
        low, high = float(low), float(high)
        if high <= low:
            raise ValueError(f"invalid SPAQ score range for {_row_id(value)}")
        if value.get("native_score") is None:
            raise ValueError(f"missing SPAQ MOS for {_row_id(value)}")
        value["technical_target"] = min(1.0, max(0.0, (float(value["native_score"]) - low) / (high - low)))
        result.append(value)
    return result


create_spaq_technical_splits = prepare_spaq_technical_splits
build_spaq_splits = prepare_spaq_technical_splits


class SPAQTechnicalDataset(Dataset):
    """SPAQ auxiliary technical task joined to immutable joined-ten features."""

    def __init__(self, split_manifest: Path, join_dir: Path, *, partition: str,
                 validation_fold: int | None = None, allow_locked_test: bool = False,
                 max_open_shards: int = 8):
        if partition not in {"train", "validation", "locked_test"}:
            raise ValueError(f"unknown partition: {partition}")
        if partition in {"train", "validation"} and validation_fold not in range(5):
            raise ValueError("development partitions require validation_fold in [0, 4]")
        if partition == "locked_test" and not allow_locked_test:
            raise PermissionError("locked test requires explicit opt-in")
        all_rows = [json.loads(line) for line in Path(split_manifest).read_text().splitlines() if line.strip()]
        rows = [row for row in all_rows if self._include(row, partition, validation_fold)]
        if any(str(row.get("dataset_id")) != "spaq" for row in rows):
            raise ValueError("SPAQ technical manifest contains a non-SPAQ row")
        self.partition, self.validation_fold = partition, validation_fold
        self.split_manifest, self.join_dir = Path(split_manifest), Path(join_dir)
        self.dataset = JoinedTenDataset(self.join_dir, max_open_shards=max_open_shards)
        indexes = {str(row["image_id"]): index for index, row in enumerate(self.dataset.rows)}
        self.samples = []
        for row in rows:
            feature_id = str(row.get("feature_id") or row["image_id"])
            if feature_id not in indexes:
                raise ValueError(f"missing joined-ten feature for spaq:{feature_id}")
            self.samples.append((indexes[feature_id], row))

    @staticmethod
    def _include(row: Mapping[str, Any], partition: str, validation_fold: int | None) -> bool:
        locked = bool(row.get("locked_test"))
        if partition == "locked_test":
            return locked
        if locked:
            return False
        fold = row.get("development_fold")
        return fold != validation_fold if partition == "train" else fold == validation_fold

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        feature_index, row = self.samples[index]
        item = self.dataset[feature_index]
        target = float(row.get("technical_target", row.get("normalized_mos", 0.0)))
        return {
            "features": item["features"], "target": target, "technical_target": target,
            "native_target": float(row["native_score"]), "dataset_id": "spaq",
            "image_id": item["image_id"], "group_id": str(row["group_id"]),
            "task": "technical", "development_fold": -1 if row.get("development_fold") is None else int(row["development_fold"]),
        }

    def fit_normalizer(self, stats_path: Path | None = None) -> FeatureNormalizer:
        if self.partition != "train":
            raise ValueError("feature normalization must be fitted on a train partition")
        identity = {
            "schema_version": "spaq-technical-view-v1",
            "split_manifest_sha256": hashlib.sha256(self.split_manifest.read_bytes()).hexdigest(),
            "partition": self.partition, "validation_fold": self.validation_fold,
            "feature_dimension": 1544, "dataset_id": "spaq",
        }
        join_digest = hashlib.sha256()
        for filename in ("join-audit.json", "join.jsonl"):
            candidate = self.join_dir / filename
            if candidate.is_file():
                join_digest.update(candidate.read_bytes())
        identity["join_hash"] = join_digest.hexdigest()
        return FeatureNormalizer.fit((self[index]["features"] for index in range(len(self))),
                                     identity=identity, stats_path=stats_path)

    def normalized(self, normalizer: FeatureNormalizer) -> NormalizedFusionTrainingDataset:
        return NormalizedFusionTrainingDataset(self, normalizer)


SPAQTechnicalTrainingDataset = SPAQTechnicalDataset
