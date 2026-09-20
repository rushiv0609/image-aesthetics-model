"""Cross-dataset deduplication and deterministic leakage-safe split preparation."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

VERSION = "fusion-prep-v3"


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {v: v for v in values}

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


def _dhash(path: Path, size: int = 16) -> int:
    with Image.open(path) as image:
        gray = ImageOps.grayscale(image).resize((size + 1, size))
        pixels = list(gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for col in range(size):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


def _distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def group_exact(records: list[dict]) -> dict[str, str]:
    """Create connected components for exact content hashes across datasets."""
    uf = UnionFind(str(row["record_id"]) for row in records)
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_hash[str(row["sha256"])].append(str(row["record_id"]))
    for members in by_hash.values():
        for member in members[1:]:
            uf.union(members[0], member)
    components: dict[str, list[str]] = defaultdict(list)
    for row in records: components[uf.find(str(row["record_id"]))].append(str(row["record_id"]))
    names = {root: "group-" + hashlib.sha256(min(members).encode()).hexdigest()[:16] for root, members in components.items()}
    return {str(row["record_id"]): names[uf.find(str(row["record_id"]))] for row in records}


def near_duplicate_candidates(records: list[dict], image_roots: dict[str, Path], threshold: int = 8, workers: int = 8) -> list[dict]:
    """Return deterministic dHash candidates for human review; never merges them."""
    inputs = [(row, image_roots[str(row["dataset_id"])] / str(row["image_name"])) for row in sorted(records, key=lambda x: str(x["record_id"]))]
    inputs = [(row, path) for row, path in inputs if path.is_file()]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dhash") as pool:
        values = list(pool.map(lambda item: _dhash(item[1]), inputs))
    hashes = list(zip((item[0] for item in inputs), values))
    # Neighboring high bits are a cheap deterministic candidate index.  We still
    # compute exact Hamming distance before emitting a candidate.
    # Split the 256-bit hash into threshold+1 disjoint bands. Any pair within
    # Hamming distance <= threshold must share at least one complete band.
    band_count = threshold + 1
    band_width = 256 // band_count
    buckets: dict[tuple[int, int], list[tuple[dict, int]]] = defaultdict(list)
    for row, value in hashes:
        for band in range(band_count):
            shift = band * band_width
            width = band_width if band < band_count - 1 else 256 - shift
            buckets[(band, (value >> shift) & ((1 << width) - 1))].append((row, value))
    candidates: dict[tuple[str, str], dict] = {}
    for row, value in hashes:
        seen: set[str] = set()
        for band in range(band_count):
            shift = band * band_width
            width = band_width if band < band_count - 1 else 256 - shift
            for other, other_value in buckets.get((band, (value >> shift) & ((1 << width) - 1)), ()):
                left, right = str(row["record_id"]), str(other["record_id"])
                if left >= right or right in seen or row["sha256"] == other["sha256"]:
                    continue
                seen.add(right)
                distance = _distance(value, other_value)
                if distance <= threshold:
                    candidates[(left, right)] = {"left": left, "right": right, "dhash_distance": distance, "review_required": True}
    return sorted(candidates.values(), key=lambda x: (x["dhash_distance"], x["left"], x["right"]))


def _bins(records: list[dict], bins: int = 10) -> dict[str, int]:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        by_dataset[str(row["dataset_id"])].append(row)
    result: dict[str, int] = {}
    for dataset, rows in by_dataset.items():
        ordered = sorted(rows, key=lambda x: (float(x["native_score"]), str(x["record_id"])))
        for index, row in enumerate(ordered):
            result[str(row["record_id"])] = min(bins - 1, index * bins // len(ordered))
    return result


def assign_global_folds(records: list[dict], groups: dict[str, str], folds: int = 5, seed: int = 20260920) -> dict[str, int]:
    """Assign each connected group once, approximately balancing bins per dataset."""
    if folds < 2:
        raise ValueError("folds must be >= 2")
    bins = _bins(records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[groups[str(row["record_id"])]].append(row)
    datasets = sorted({str(row["dataset_id"]) for row in records})
    target: dict[tuple[str, int], float] = {}
    target_rows: dict[str, float] = {}
    totals: dict[str, int] = defaultdict(int)
    for row in records:
        totals[str(row["dataset_id"])] += 1
    for dataset in datasets:
        target_rows[dataset] = totals[dataset] / folds
        for b in range(10):
            target[(dataset, b)] = sum(1 for row in records if str(row["dataset_id"]) == dataset and bins[str(row["record_id"])] == b) / folds
    buckets_groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for group, members in grouped.items():
        # A shared cross-dataset group is keyed by its first stable member;
        # keeping it intact is more important than forcing exact per-dataset
        # quotas for that rare case.
        anchor = min(members, key=lambda r: (str(r["dataset_id"]), str(r["record_id"])))
        buckets_groups[(str(anchor["dataset_id"]), bins[str(anchor["record_id"])])].append(group)
    ordered: list[str] = []
    for key in sorted(buckets_groups):
        ordered.extend(sorted(buckets_groups[key], key=lambda g: (hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), g)))
    counts: dict[tuple[int, str, int], int] = defaultdict(int)
    assignment: dict[str, int] = {}
    bucket_positions: dict[tuple[str, int], int] = defaultdict(int)
    for position, group in enumerate(ordered):
        members = grouped[group]
        costs = []
        for fold in range(folds):
            before = 0.0
            after = 0.0
            additions: dict[tuple[str, int], int] = defaultdict(int)
            for row in members:
                additions[(str(row["dataset_id"]), bins[str(row["record_id"])])] += 1
            for key, amount in additions.items():
                current = counts[(fold, *key)]
                denom = max(1.0, target[key])
                before += ((current - target[key]) / denom) ** 2
                after += ((current + amount - target[key]) / denom) ** 2
            for dataset in datasets:
                current_rows = sum(counts[(fold, dataset, b)] for b in range(10))
                added_rows = sum(amount for (ds, _), amount in additions.items() if ds == dataset)
                denom = max(1.0, target_rows[dataset])
                before += 2.0 * ((current_rows - target_rows[dataset]) / denom) ** 2
                after += 2.0 * ((current_rows + added_rows - target_rows[dataset]) / denom) ** 2
            costs.append(after - before)
        # Seed each fold before optimizing; this prevents the normalized
        # quantile objective from leaving tail folds empty on small corpora.
        anchor = min(members, key=lambda r: (str(r["dataset_id"]), str(r["record_id"])))
        key = (str(anchor["dataset_id"]), bins[str(anchor["record_id"])])
        bucket_index = bucket_positions[key]
        fold = bucket_index if bucket_index < folds else min(range(folds), key=lambda i: (costs[i], i))
        bucket_positions[key] += 1
        for row in members:
            record_id = str(row["record_id"])
            assignment[record_id] = fold
            counts[(fold, str(row["dataset_id"]), bins[record_id])] += 1
    return assignment


def prepare_fusion(records: list[dict], image_roots: dict[str, Path], *, folds: int = 5, seed: int = 20260920) -> tuple[list[dict], list[dict]]:
    groups = group_exact(records)
    near = near_duplicate_candidates(records, image_roots)
    exact_group_count = len(set(groups.values()))
    quarantined = {x for c in near for x in (c["left"], c["right"])}
    # Preserve exact components while adding only declared source-group edges.
    uf = UnionFind(groups)
    members_by_label: dict[str, list[str]] = defaultdict(list)
    for rid, label in groups.items(): members_by_label[label].append(rid)
    for members in members_by_label.values():
        for member in members[1:]: uf.union(members[0], member)
    source_keys: dict[str, str] = {}
    for row in records:
        for key in row.get("source_groups", ()):
            source_keys.setdefault(str(key), str(row["record_id"]))
            uf.union(source_keys[str(key)], str(row["record_id"]))
    assert len(set(groups.values())) == exact_group_count
    roots = {uf.find(record_id) for record_id in groups}
    members: dict[str, list[str]] = defaultdict(list)
    for record_id in groups: members[uf.find(record_id)].append(record_id)
    names = {root: "group-" + hashlib.sha256(min(values).encode()).hexdigest()[:16] for root, values in members.items()}
    quarantined_roots = {uf.find(record_id) for record_id in quarantined}
    quarantined = {record_id for record_id in groups if uf.find(record_id) in quarantined_roots}
    groups = {record_id: names[uf.find(record_id)] for record_id in groups if record_id not in quarantined}
    records = [row for row in records if str(row["record_id"]) not in quarantined]
    bins = _bins(records)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[groups[str(row["record_id"])]].append(row)
    # Select a group-safe locked test set by dataset/bin quotas. Groups shared
    # across datasets are indivisible; deterministic hash order resolves ties.
    targets = {(d, b): round(sum(1 for r in records if r["dataset_id"] == d and bins[r["record_id"]] == b) * 0.1) for d in {r["dataset_id"] for r in records} for b in range(10)}
    selected = defaultdict(int)
    locked_groups: set[str] = set()
    for group in sorted(grouped, key=lambda g: (hashlib.sha256(f"test:{seed}:{g}".encode()).hexdigest(), g)):
        members = grouped[group]
        contribution = defaultdict(int)
        for row in members: contribution[(row["dataset_id"], bins[row["record_id"]])] += 1
        if all(selected[key] + amount <= targets[key] for key, amount in contribution.items()):
            locked_groups.add(group)
            for key, amount in contribution.items(): selected[key] += amount
    development = [row for row in records if groups[str(row["record_id"])] not in locked_groups]
    folds_by_id = assign_global_folds(development, {k: v for k, v in groups.items() if v not in locked_groups}, folds=folds, seed=seed)
    rows = []
    for row in sorted(records, key=lambda x: (str(x["dataset_id"]), str(x["image_id"]))):
        record = dict(row)
        record["record_id"] = str(row["dataset_id"]) + ":" + str(row["image_id"])
        record["group_id"] = groups[record["record_id"]]
        record["development_fold"] = None if record["record_id"] not in folds_by_id else folds_by_id[record["record_id"]]
        record["locked_test"] = groups[record["record_id"]] in locked_groups
        rows.append(record)
    return rows, near


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
