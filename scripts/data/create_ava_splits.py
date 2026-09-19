#!/usr/bin/env python3
"""Materialize deterministic AVA outer and development-CV split manifests."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.splits import (  # noqa: E402
    ALGORITHM_VERSION,
    DEFAULT_SEED,
    assign_cv,
    assign_outer,
    completed_records,
    connected_group_keys,
    exact_duplicate_groups,
    partition_rows,
    sha256_file,
    sha256_json,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, rows: list[dict]) -> str:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
    temp.replace(path)
    return sha256_file(path)


def row_stats(rows: list[dict], global_deciles: dict[str, int]) -> dict:
    means = np.asarray([float(row["mean_rating"]) for row in rows], dtype=np.float64)
    deciles = np.asarray([global_deciles[str(row["image_id"])] for row in rows], dtype=np.int8)
    return {
        "rows": len(rows),
        "mean_rating": {"mean": float(means.mean()), "std": float(means.std()),
                        "p05": float(np.quantile(means, .05)), "p50": float(np.quantile(means, .50)),
                        "p95": float(np.quantile(means, .95))},
        "decile_counts": np.bincount(deciles, minlength=10).astype(int).tolist(),
        "vote_total_mean": float(np.mean([sum(row["vote_counts"]) for row in rows])),
        "extremes": {"mean_lt_3": int((means < 3).sum()), "mean_ge_8": int((means >= 8).sum()),
                     "mean_ge_8_5": int((means >= 8.5).sum())},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-audit-json", type=Path, required=True)
    parser.add_argument("--repo-audit-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hash-workers", type=int, default=8)
    args = parser.parse_args()

    source_rows = read_jsonl(args.manifest)
    by_id = {str(row["image_id"]): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("full manifest contains duplicate image IDs")
    con = sqlite3.connect(args.ledger)
    completed = {str(row[0]) for row in con.execute("SELECT image_id FROM items WHERE state='completed'")}
    ledger_ids = {str(row[0]) for row in con.execute("SELECT image_id FROM items")}
    con.close()
    if ledger_ids != set(by_id):
        raise ValueError("cache ledger membership differs from full manifest")
    records, excluded = completed_records(source_rows, completed)

    duplicate_groups = exact_duplicate_groups(source_rows, args.hash_workers)
    group_keys = connected_group_keys(records, duplicate_groups)
    outer = assign_outer(records, group_keys, args.seed)
    development = [row for row in records if outer[str(row["image_id"])] != "test"]
    cv = assign_cv(development, {str(row["image_id"]): group_keys[str(row["image_id"])] for row in development}, 5, args.seed)
    from image_aesthetics.data.splits import rating_deciles
    global_decile_values = rating_deciles(records)
    global_deciles = {str(row["image_id"]): int(global_decile_values[index]) for index, row in enumerate(records)}

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized = {}
    for partition in ("train", "validation", "test"):
        rows = partition_rows(records, outer, partition)
        output_rows = [{"image_id": str(row["image_id"]), "manifest_position": int(row["manifest_position"]),
                        "challenge_id": int(row["challenge_id"]), "group_id": group_keys[str(row["image_id"])],
                        "mean_rating": float(row["mean_rating"]), "normalized_mean": float(row["normalized_mean"]),
                        "vote_counts": row["vote_counts"], "vote_total": int(row["vote_total"]),
                        "locked": partition == "test"} for row in rows]
        path = output_dir / ("locked-test.jsonl" if partition == "test" else f"{partition}.jsonl")
        materialized[partition] = {"path": path.name, "rows": len(output_rows), "sha256": atomic_jsonl(path, output_rows)}
    dev_rows = [{"image_id": str(row["image_id"]), "manifest_position": int(row["manifest_position"]),
                 "challenge_id": int(row["challenge_id"]), "group_id": group_keys[str(row["image_id"])],
                 "mean_rating": float(row["mean_rating"]), "normalized_mean": float(row["normalized_mean"]),
                 "vote_counts": row["vote_counts"], "vote_total": int(row["vote_total"])} for row in development]
    materialized["development"] = {"path": "development.jsonl", "rows": len(dev_rows), "sha256": atomic_jsonl(output_dir / "development.jsonl", dev_rows)}
    cv_rows = [{"image_id": str(row["image_id"]), "manifest_position": int(row["manifest_position"]),
                "group_id": group_keys[str(row["image_id"])], "fold": cv[str(row["image_id"])]} for row in development]
    materialized["cv_assignments"] = {"path": "cv-assignments.jsonl", "rows": len(cv_rows), "sha256": atomic_jsonl(output_dir / "cv-assignments.jsonl", cv_rows)}

    outer_stats = {partition: row_stats(partition_rows(records, outer, partition), global_deciles) for partition in ("train", "validation", "test")}
    cv_stats = {str(fold): row_stats([row for row in development if cv[str(row["image_id"])] == fold], global_deciles) for fold in range(5)}
    cache_metadata = json.loads(args.cache_manifest.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1, "algorithm_version": ALGORITHM_VERSION, "seed": args.seed,
        "dataset_manifest_sha256": sha256_file(args.manifest), "dataset_manifest_rows": len(source_rows),
        "completed_cache_rows": len(records), "excluded_image_ids_count": len(excluded),
        "excluded_image_ids_sha256": sha256_json(excluded), "cache_manifest_sha256": sha256_file(args.cache_manifest),
        "model_revision": cache_metadata["contract"]["model_revision"],
        "cache_dataset_fingerprint": cache_metadata.get("dataset_fingerprint"),
        "duplicate_group_count": len(duplicate_groups), "group_count": len(set(group_keys.values())),
        "outer": materialized, "outer_stats": outer_stats, "cv_stats": cv_stats,
        "locked_test": {"path": materialized["test"]["path"], "requires_explicit_opt_in": True},
    }
    (output_dir / "split-metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {"schema_version": 1, "algorithm_version": ALGORITHM_VERSION, "seed": args.seed,
             "dataset_manifest_rows": len(source_rows), "completed_cache_rows": len(records),
             "excluded_rows": len(excluded), "duplicate_group_count": len(duplicate_groups),
             "group_count": len(set(group_keys.values())), "outer": materialized,
             "outer_stats": outer_stats, "cv_stats": cv_stats,
             "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
             "cache_manifest_sha256": metadata["cache_manifest_sha256"],
             "cache_dataset_fingerprint": metadata["cache_dataset_fingerprint"],
             "model_revision": metadata["model_revision"],
             "split_metadata_sha256": sha256_file(output_dir / "split-metadata.json")}
    args.repo_audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.repo_audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# AVA materialized split audit", "", f"Algorithm: `{ALGORITHM_VERSION}`; seed: `{args.seed}`.", "",
             f"The full manifest contains {len(source_rows):,} rows; the completed embedding ledger contributes {len(records):,}. The remaining {len(excluded)} rows are excluded from all partitions.", "",
             "| Partition | Rows | Groups | Mean | Std | Decile counts |", "|:--|--:|--:|--:|--:|:--|"]
    for name in ("train", "validation", "test"):
        stats = outer_stats[name]; lines.append(f"| {name}{' (locked)' if name == 'test' else ''} | {stats['rows']:,} | {len({row['group_id'] for row in (read_jsonl(output_dir / ('locked-test.jsonl' if name == 'test' else f'{name}.jsonl')))})} | {stats['mean_rating']['mean']:.4f} | {stats['mean_rating']['std']:.4f} | `{stats['decile_counts']}` |")
    lines += ["", "The locked test requires explicit opt-in and is not available to normal training or hyperparameter-selection code. Five group-disjoint CV fold assignments are materialized only over the development rows.", "", "Dataset/cache/model/split hashes and aggregate CV statistics are in the external `split-metadata.json` and repository JSON audit.", ""]
    args.repo_audit_md.parent.mkdir(parents=True, exist_ok=True)
    args.repo_audit_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed": len(records), "excluded": len(excluded), "groups": len(set(group_keys.values())), "output_dir": str(output_dir), "audit": str(args.repo_audit_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
