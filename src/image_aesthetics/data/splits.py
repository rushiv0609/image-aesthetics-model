"""Deterministic leakage-resistant AVA split construction and guarded loading."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np


ALGORITHM_VERSION = "ava-group-stratified-v1"
DEFAULT_SEED = 20260920
PARTITIONS = ("train", "validation", "test")
FRACTIONS = np.asarray((0.8, 0.1, 0.1), dtype=np.float64)


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rating_deciles(records: list[dict]) -> np.ndarray:
    means = np.asarray([float(row["mean_rating"]) for row in records], dtype=np.float64)
    order = np.argsort(means, kind="stable")
    deciles = np.empty(len(records), dtype=np.int8)
    deciles[order] = np.minimum(9, np.arange(len(records), dtype=np.int64) * 10 // len(records))
    return deciles


def exact_duplicate_groups(records: list[dict], workers: int = 8) -> list[list[str]]:
    """Hash only same-size candidates, returning deterministic exact groups."""
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in records:
        by_size[int(row["source_size"])].append(row)
    candidates = [row for group in by_size.values() if len(group) > 1 for row in group]

    def digest(row: dict) -> tuple[str, str]:
        return str(row["image_id"]), sha256_file(Path(row["source_path"]))

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ava-duplicate-hash") as pool:
        hashes = list(pool.map(digest, candidates))
    grouped: dict[str, list[str]] = defaultdict(list)
    for image_id, digest in hashes:
        grouped[digest].append(image_id)
    return sorted((sorted(group) for group in grouped.values() if len(group) > 1), key=lambda group: group[0])


def connected_group_keys(records: list[dict], duplicate_groups: list[list[str]]) -> dict[str, str]:
    """Union all challenge members and exact duplicate relationships."""
    ids = [str(row["image_id"]) for row in records]
    uf = UnionFind(ids)
    challenges: dict[str, list[str]] = defaultdict(list)
    for row in records:
        challenges[str(row["challenge_id"])].append(str(row["image_id"]))
    for group in challenges.values():
        for image_id in group[1:]:
            uf.union(group[0], image_id)
    present = set(ids)
    for group in duplicate_groups:
        # Duplicate groups are recomputed against the full manifest, while
        # the split rows intentionally contain only completed cache items.
        # Ignore failed/omitted members rather than allowing them to enter the
        # union-find domain.
        members = [image_id for image_id in group if image_id in present]
        for image_id in members[1:]:
            uf.union(members[0], image_id)
    roots = {uf.find(image_id) for image_id in ids}
    names = {root: f"group-{hashlib.sha256(root.encode()).hexdigest()[:16]}" for root in roots}
    return {image_id: names[uf.find(image_id)] for image_id in ids}


def _greedy_assignment(records: list[dict], group_keys: dict[str, str], fractions: np.ndarray,
                       seed: int, names: tuple[str, ...]) -> dict[str, str]:
    deciles = rating_deciles(records)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        grouped[group_keys[str(row["image_id"]) ]].append(index)
    global_hist = np.bincount(deciles, minlength=10).astype(np.float64)
    target_rows = fractions * len(records)
    target_hist = fractions[:, None] * global_hist[None, :]
    rng = random.Random(seed)
    groups = list(grouped)
    groups.sort(key=lambda key: (-len(grouped[key]), rng.random(), key))
    counts = np.zeros(len(names), dtype=np.float64)
    hist = np.zeros((len(names), 10), dtype=np.float64)
    assignment: dict[str, str] = {}
    for key in groups:
        indexes = grouped[key]
        group_hist = np.bincount(deciles[indexes], minlength=10).astype(np.float64)
        costs: list[float] = []
        for split in range(len(names)):
            new_counts = counts.copy(); new_counts[split] += len(indexes)
            new_hist = hist.copy(); new_hist[split] += group_hist
            row_error = np.sum(((new_counts - target_rows) / np.maximum(1.0, target_rows)) ** 2)
            # Row proportions are the primary objective.  A small decile
            # penalty preserves label balance without making a large group
            # permanently ineligible for a small target partition (which is
            # important for deterministic tests and grouped real data).
            hist_error = np.sum(((new_hist - target_hist) / np.maximum(1.0, target_hist)) ** 2)
            costs.append(float(row_error + 0.02 * hist_error))
        split = int(np.argmin(costs))
        counts[split] += len(indexes); hist[split] += group_hist
        for index in indexes:
            assignment[str(records[index]["image_id"])] = names[split]
    return assignment


def assign_outer(records: list[dict], group_keys: dict[str, str], seed: int = DEFAULT_SEED) -> dict[str, str]:
    return _greedy_assignment(records, group_keys, FRACTIONS, seed, PARTITIONS)


def assign_cv(records: list[dict], group_keys: dict[str, str], folds: int = 5,
              seed: int = DEFAULT_SEED) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    names = tuple(str(index) for index in range(folds))
    fractions = np.full(folds, 1.0 / folds, dtype=np.float64)
    assignment = _greedy_assignment(records, group_keys, fractions, seed, names)
    return {image_id: int(fold) for image_id, fold in assignment.items()}


def partition_rows(records: list[dict], outer: dict[str, str], partition: str) -> list[dict]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    return [row for row in records if outer[str(row["image_id"])] == partition]


def completed_records(records: list[dict], completed_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Validate ledger membership and return completed rows plus excluded IDs."""
    manifest_ids = {str(row["image_id"]) for row in records}
    unknown = completed_ids - manifest_ids
    if unknown:
        raise ValueError(f"cache ledger contains {len(unknown)} IDs absent from the manifest")
    by_id = {str(row["image_id"]): row for row in records}
    completed = [by_id[image_id] for image_id in sorted(completed_ids, key=lambda value: int(by_id[value]["manifest_position"]))]
    excluded = sorted(manifest_ids - completed_ids, key=lambda value: int(by_id[value]["manifest_position"]))
    return completed, excluded


def load_partition(path: Path, partition: str, *, allow_locked_test: bool = False) -> list[dict]:
    """Load one materialized partition; locked test requires explicit opt-in."""
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition: {partition}")
    if partition == "test" and not allow_locked_test:
        raise PermissionError("locked AVA test partition is unavailable to normal training/hparam code")
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
