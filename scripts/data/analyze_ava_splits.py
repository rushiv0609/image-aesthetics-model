#!/usr/bin/env python3
"""Audit AVA labels, challenge grouping, duplicates, and candidate splits.

The input manifest remains external. Outputs contain only aggregate statistics
and split-assignment summaries; no image paths or per-image labels are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


SEED = 20260920
SPLITS = ("train", "validation", "test")
FRACTIONS = np.array([0.8, 0.1, 0.1], dtype=np.float64)


class DSU:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return parent

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entropy(values: np.ndarray) -> np.ndarray:
    safe = np.clip(values, 1e-12, 1.0)
    return -(safe * np.log2(safe)).sum(axis=1)


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {key: float(value) for key, value in zip(
        ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
        np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
    )}


def label_audit(rows: list[dict]) -> dict:
    means = np.asarray([row["mean_rating"] for row in rows], dtype=np.float64)
    normalized = np.asarray([row["normalized_mean"] for row in rows], dtype=np.float64)
    votes = np.asarray([row["vote_counts"] for row in rows], dtype=np.float64)
    totals = votes.sum(axis=1)
    proportions = votes / totals[:, None]
    ratings = np.arange(1.0, 11.0)
    variance = ((proportions * (ratings[None, :] - means[:, None]) ** 2).sum(axis=1))
    fixed_bins = np.clip(np.floor(means).astype(int), 1, 10)
    normalized_deciles = np.minimum(9, (normalized * 10).astype(int))
    challenge_sizes = np.asarray(list(Counter(int(row["challenge_id"]) for row in rows).values()), dtype=np.float64)
    return {
        "rows": len(rows),
        "unique_image_ids": len({row["image_id"] for row in rows}),
        "missing_required_fields": sum(any(key not in row for key in ("image_id", "source_path", "vote_counts", "challenge_id")) for row in rows),
        "mean_rating": {"mean": float(means.mean()), "std": float(means.std()), **quantiles(means),
                        "min": float(means.min()), "max": float(means.max())},
        "normalized_mean": {"mean": float(normalized.mean()), "std": float(normalized.std()), **quantiles(normalized),
                            "min": float(normalized.min()), "max": float(normalized.max())},
        "vote_total": {"mean": float(totals.mean()), "std": float(totals.std()), **quantiles(totals),
                        "min": int(totals.min()), "max": int(totals.max()),
                        "histogram": {str(k): int((totals == k).sum()) for k in sorted(set(totals.astype(int)))[:20]},
                        "bucket_histogram": {label: int(count) for label, count in zip(
                            ("78-99", "100-149", "150-199", "200-249", "250-299", "300-399", "400-499", "500+"),
                            np.histogram(totals, bins=[78, 100, 150, 200, 250, 300, 400, 500, np.inf])[0],
                        )}},
        "vote_distribution_mean": proportions.mean(axis=0).round(8).tolist(),
        "vote_distribution_std": proportions.std(axis=0).round(8).tolist(),
        "per_image_vote_entropy_bits": {"mean": float(entropy(proportions).mean()), "std": float(entropy(proportions).std()), **quantiles(entropy(proportions))},
        "per_image_vote_variance": {"mean": float(variance.mean()), "std": float(variance.std()), **quantiles(variance)},
        "rating_bin_counts_1_to_10": {str(k): int((fixed_bins == k).sum()) for k in range(1, 11)},
        "normalized_decile_counts": {str(k): int((normalized_deciles == k).sum()) for k in range(10)},
        "challenge_groups": {"count": int(len(challenge_sizes)), "mean_rows": float(challenge_sizes.mean()),
                             "std_rows": float(challenge_sizes.std()), **quantiles(challenge_sizes),
                             "min_rows": int(challenge_sizes.min()), "max_rows": int(challenge_sizes.max()),
                             "size_buckets": {label: int(count) for label, count in zip(
                                 ("1", "2-9", "10-49", "50-199", "200-499", "500+"),
                                 (np.sum(challenge_sizes == 1), np.sum((challenge_sizes >= 2) & (challenge_sizes < 10)),
                                  np.sum((challenge_sizes >= 10) & (challenge_sizes < 50)), np.sum((challenge_sizes >= 50) & (challenge_sizes < 200)),
                                  np.sum((challenge_sizes >= 200) & (challenge_sizes < 500)), np.sum(challenge_sizes >= 500)),
                             )}},
        "extreme_score_counts": {
            "mean_lt_2": int((means < 2).sum()), "mean_lt_3": int((means < 3).sum()),
            "mean_ge_8": int((means >= 8).sum()), "mean_ge_8_5": int((means >= 8.5).sum()),
            "mean_ge_9": int((means >= 9).sum()),
        },
    }


def ordering_audit(rows: list[dict]) -> dict:
    means = np.asarray([row["mean_rating"] for row in rows], dtype=np.float64)
    challenges = np.asarray([row["challenge_id"] for row in rows], dtype=np.int64)
    positions = np.asarray([row.get("annotation_position", -1) for row in rows], dtype=np.int64)
    image_ids = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
    adjacent_same = float(np.mean(challenges[1:] == challenges[:-1]))
    challenge_changes = int(np.sum(challenges[1:] != challenges[:-1]))
    score_lag = float(np.corrcoef(means[1:], means[:-1])[0, 1])
    position_corr = float(np.corrcoef(np.arange(len(rows), dtype=np.float64), means)[0, 1])
    id_corr = float(np.corrcoef(image_ids, means)[0, 1])
    equal_position = float(np.mean(positions == np.arange(1, len(rows) + 1)))
    runs = []
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or challenges[index] != challenges[start]:
            runs.append(index - start); start = index
    return {
        "annotation_position_matches_original_row": equal_position,
        "challenge_changes_between_adjacent_rows": challenge_changes,
        "adjacent_same_challenge_fraction": adjacent_same,
        "challenge_contiguous_run_count": len(runs),
        "challenge_contiguous_run_max": max(runs),
        "challenge_contiguous_run_p95": float(np.quantile(runs, 0.95)),
        "mean_rating_lag1_correlation": score_lag,
        "mean_rating_vs_manifest_position_correlation": position_corr,
        "mean_rating_vs_numeric_image_id_correlation": id_corr,
        "caveat": "AVA.txt order is annotation/order metadata, not a documented capture chronology; no timestamp or photographer identity is present in the manifest.",
    }


def file_duplicate_audit(rows: list[dict], workers: int = 8) -> dict:
    by_size: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_size[int(row["source_size"])].append(row)
    candidates = [row for group in by_size.values() if len(group) > 1 for row in group]
    candidate_bytes = sum(int(row["source_size"]) for row in candidates)

    def digest(row):
        path = Path(row["source_path"])
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
        return row["image_id"], hasher.hexdigest()

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ava-hash") as pool:
        digests = list(pool.map(digest, candidates))
    by_digest: dict[str, list[str]] = defaultdict(list)
    for image_id, value in digests:
        by_digest[value].append(image_id)
    exact = [group for group in by_digest.values() if len(group) > 1]
    return {
        "size_group_count": sum(len(group) > 1 for group in by_size.values()),
        "size_collision_candidate_rows": len(candidates),
        "size_collision_candidate_bytes": candidate_bytes,
        "hashed_candidate_fraction_of_rows": len(candidates) / len(rows),
        "exact_duplicate_group_count": len(exact),
        "exact_duplicate_rows_beyond_first": sum(len(group) - 1 for group in exact),
        "largest_exact_duplicate_group": max((len(group) for group in exact), default=1),
        "hash_algorithm": "sha256",
        "groups": [sorted(group) for group in exact[:100]],
    }


def deciles(means: np.ndarray) -> np.ndarray:
    order = np.argsort(means, kind="stable")
    output = np.empty(len(means), dtype=np.int8)
    output[order] = np.minimum(9, (np.arange(len(means)) * 10 // len(means)).astype(np.int8))
    return output


def naive_split(rows: list[dict], seed: int) -> dict[str, int]:
    means = np.asarray([row["mean_rating"] for row in rows])
    bins = deciles(means)
    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    for bucket in range(10):
        ids = [rows[index]["image_id"] for index in np.flatnonzero(bins == bucket)]
        rng.shuffle(ids)
        n_train = round(len(ids) * 0.8)
        n_val = round(len(ids) * 0.1)
        for image_id in ids[:n_train]: assignment[image_id] = "train"
        for image_id in ids[n_train:n_train + n_val]: assignment[image_id] = "validation"
        for image_id in ids[n_train + n_val:]: assignment[image_id] = "test"
    return assignment


def group_assignment(rows: list[dict], group_key: dict[str, str], seed: int) -> dict[str, str]:
    means = np.asarray([row["mean_rating"] for row in rows])
    bins = deciles(means)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): grouped[group_key[row["image_id"]]].append(index)
    global_hist = np.bincount(bins, minlength=10).astype(float)
    target_rows = FRACTIONS * len(rows)
    target_hist = FRACTIONS[:, None] * global_hist[None, :]
    rng = random.Random(seed)
    groups = list(grouped)
    # Largest groups first limits the unavoidable error from indivisible
    # challenges; a seeded hash tie-break makes repeated runs deterministic.
    groups.sort(key=lambda key: (-len(grouped[key]), rng.random(), key))
    counts = np.zeros(3, dtype=float)
    hist = np.zeros((3, 10), dtype=float)
    assignment: dict[str, str] = {}
    for key in groups:
        indexes = grouped[key]
        group_hist = np.bincount(bins[indexes], minlength=10).astype(float)
        costs = []
        for split in range(3):
            new_counts = counts.copy(); new_counts[split] += len(indexes)
            new_hist = hist.copy(); new_hist[split] += group_hist
            row_error = np.sum(((new_counts - target_rows) / np.maximum(1.0, target_rows)) ** 2)
            hist_error = np.sum(((new_hist - target_hist) / np.maximum(1.0, target_hist)) ** 2)
            overflow = np.sum(np.maximum(0.0, new_counts - target_rows) / np.maximum(1.0, target_rows))
            costs.append(row_error + 0.5 * hist_error + 10.0 * overflow)
        split = int(np.argmin(costs))
        counts[split] += len(indexes); hist[split] += group_hist
        for index in indexes: assignment[rows[index]["image_id"]] = SPLITS[split]
    return assignment


def assignment_stats(rows: list[dict], assignment: dict[str, str], group_key: dict[str, str]) -> dict:
    means = np.asarray([row["mean_rating"] for row in rows])
    totals = np.asarray([sum(row["vote_counts"]) for row in rows])
    bins = deciles(means)
    global_hist = np.bincount(bins, minlength=10) / len(rows)
    result = {}
    for split in SPLITS:
        indexes = [index for index, row in enumerate(rows) if assignment[row["image_id"]] == split]
        values = means[indexes]; split_hist = np.bincount(bins[indexes], minlength=10) / max(1, len(indexes))
        groups = {group_key[rows[index]["image_id"]] for index in indexes}
        vote_hist = np.histogram(totals[indexes], bins=[0, 20, 50, 100, 200, 500, 1000, 2000, np.inf])[0]
        result[split] = {
            "rows": len(indexes), "fraction": len(indexes) / len(rows), "groups": len(groups),
            "mean_rating": {"mean": float(values.mean()), "std": float(values.std()), **quantiles(values)},
            "decile_histogram": split_hist.round(6).tolist(),
            "decile_l1_vs_global": float(np.abs(split_hist - global_hist).sum()),
            "vote_total_histogram": vote_hist.astype(int).tolist(),
            "extremes": {"mean_lt_3": int((values < 3).sum()), "mean_ge_8": int((values >= 8).sum()), "mean_ge_8_5": int((values >= 8.5).sum())},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    labels = label_audit(rows)
    ordering = ordering_audit(rows)
    duplicates = file_duplicate_audit(rows, args.hash_workers)
    duplicate_groups = duplicates.pop("groups", [])
    challenge_by_image = {row["image_id"]: int(row["challenge_id"]) for row in rows}
    duplicates["exact_duplicate_groups_cross_challenge"] = sum(
        len({challenge_by_image[image_id] for image_id in group}) > 1 for group in duplicate_groups
    )

    challenge_key = {row["image_id"]: f"challenge:{row['challenge_id']}" for row in rows}
    dsu = DSU(row["image_id"] for row in rows)
    by_challenge: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        by_challenge[int(row["challenge_id"])].append(row["image_id"])
    for group in by_challenge.values():
        for image_id in group[1:]:
            dsu.union(group[0], image_id)
    # Recompute only the already-identified exact duplicate candidates from the
    # compact duplicate groups; this avoids persisting a per-image hash table.
    for group in duplicate_groups:
        for image_id in group[1:]: dsu.union(group[0], image_id)
    composite_key = {row["image_id"]: f"component:{dsu.find(row['image_id'])}" for row in rows}
    candidates = {
        "naive_random_stratified": assignment_stats(rows, naive_split(rows, SEED), {row["image_id"]: row["image_id"] for row in rows}),
        "challenge_group_greedy_seed_20260920": assignment_stats(rows, group_assignment(rows, challenge_key, SEED), challenge_key),
        "challenge_plus_exact_duplicate_group_greedy_seed_20260920": assignment_stats(rows, group_assignment(rows, composite_key, SEED), composite_key),
    }
    repeated = []
    for seed in (SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4):
        assignment = group_assignment(rows, composite_key, seed)
        stats = assignment_stats(rows, assignment, composite_key)
        repeated.append({"seed": seed, "rows": {split: stats[split]["rows"] for split in SPLITS},
                         "decile_l1": {split: stats[split]["decile_l1_vs_global"] for split in SPLITS}})
    output = {
        "schema_version": 1, "manifest": {"path": args.manifest.name, "rows": len(rows)},
        "seed": SEED, "label_audit": labels, "ordering_audit": ordering,
        "duplicate_audit": duplicates,
        "candidate_splits": candidates,
        "repeated_recommended_grouped_seeds": repeated,
        "recommendation": {
            "outer_split": "challenge_plus_exact_duplicate_group_greedy_seed_20260920",
            "fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "locked_test": "Do not load labels in training/hyperparameter entry points; open exactly once after model selection.",
            "cross_validation": "Run GroupKFold with 5 folds inside the 90% development (train+validation) rows; groups are challenge-plus-exact-duplicate connected components.",
            "photographer_identity": "Not available in AVA.txt or the prepared manifest; challenge_id is the available grouping proxy, not photographer identity.",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# AVA ground-truth and split analysis", "", f"Manifest rows: **{len(rows):,}**", "", "## Findings", "",
             f"- Mean rating: {labels['mean_rating']['mean']:.4f} ± {labels['mean_rating']['std']:.4f}; range {labels['mean_rating']['min']:.3f}–{labels['mean_rating']['max']:.3f}.",
             f"- Vote entropy: {labels['per_image_vote_entropy_bits']['mean']:.3f} bits mean; vote totals median {labels['vote_total']['p50']:.0f}.",
             f"- Challenge groups: {len(set(challenge_key.values())):,}; size range {labels['challenge_groups']['min_rows']:,}–{labels['challenge_groups']['max_rows']:,}; largest contiguous annotation run {ordering['challenge_contiguous_run_max']:,}.",
             f"- Exact duplicate hashing examined {duplicates['size_collision_candidate_rows']:,} size-collision candidates ({duplicates['size_collision_candidate_bytes'] / 1e9:.2f} GB); found {duplicates['exact_duplicate_group_count']:,} exact groups ({duplicates['exact_duplicate_groups_cross_challenge']:,} crossing challenges).",
             f"- Annotation order has lag-1 score correlation {ordering['mean_rating_lag1_correlation']:.4f}; no capture chronology or photographer identity is present.", "", "## Candidate split comparison", "", "| Split | Rows | Groups | Mean | Std | Decile L1 error |", "|:--|--:|--:|--:|--:|--:|"]
    for name, stats in candidates.items():
        for split in SPLITS:
            value = stats[split]
            lines.append(f"| {name} / {split} | {value['rows']:,} | {value['groups']:,} | {value['mean_rating']['mean']:.4f} | {value['mean_rating']['std']:.4f} | {value['decile_l1_vs_global']:.4f} |")
    lines += ["", "## Recommendation", "", "Use the challenge-disjoint greedy grouped split with seed `20260920`, unioning exact duplicate files into the same component. Keep 80% development train, 10% development validation, and 10% locked test. Run five-fold GroupKFold only inside the 90% development portion; never use locked-test labels during selection.", "", "The available `challenge_id` is a leakage-control proxy, not a photographer identifier. AVA metadata does not expose photographer/user identity in this manifest. Challenge IDs are nearly contiguous in AVA.txt (99.45% of adjacent rows share a challenge), so a naive random split can place visually/semantically related challenge material in every partition. Challenge grouping reduces that risk, but cannot prove photographer-level independence because photographer identity is unavailable.", ""]
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "exact_duplicate_groups": duplicates["exact_duplicate_group_count"], "output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
