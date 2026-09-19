#!/usr/bin/env python3
"""Prepare deterministic AVA JSONL manifests from the official AVA.txt file.

The output is intentionally JSONL so the embedding cache can stream/inspect it
without a dataframe dependency.  ``--limit`` creates a deterministic,
mean-rating-stratified smoke manifest; without it, all usable annotated images
are emitted in AVA annotation order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_row(line: str, line_number: int) -> dict[str, Any]:
    fields = line.split()
    if len(fields) != 15:
        raise ValueError(f"line {line_number}: expected 15 columns, got {len(fields)}")
    try:
        position = int(fields[0])
        image_id = str(int(fields[1]))
        votes = [int(value) for value in fields[2:12]]
        challenge_id = int(fields[14])
    except ValueError as exc:
        raise ValueError(f"line {line_number}: non-integer AVA field") from exc
    if position < 1 or any(value < 0 for value in votes) or sum(votes) <= 0:
        raise ValueError(f"line {line_number}: invalid position/votes")
    total = sum(votes)
    mean = sum((index + 1) * count for index, count in enumerate(votes)) / total
    return {
        "annotation_position": position,
        "image_id": image_id,
        "vote_counts": votes,
        "vote_total": total,
        "mean_rating": mean,
        "normalized_mean": (mean - 1.0) / 9.0,
        "challenge_id": challenge_id,
    }


def load_rows(annotation: Path, image_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    seen: set[str] = set()
    with annotation.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = parse_row(raw, line_number)
            except ValueError:
                rejects["invalid_annotation"] += 1
                continue
            image_path = image_dir / f"{row['image_id']}.jpg"
            if row["image_id"] in seen:
                rejects["duplicate_image_id"] += 1
                continue
            seen.add(row["image_id"])
            if not image_path.is_file():
                rejects["missing_image"] += 1
                continue
            stat = image_path.stat()
            if stat.st_size <= 0:
                rejects["empty_image"] += 1
                continue
            row.update({
                "source_path": str(image_path.resolve()),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
            })
            rows.append(row)
    return rows, dict(rejects)


def stratified_sample(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    # Stable score deciles, then deterministic seeded selection within each bin.
    ordered = sorted(rows, key=lambda row: (row["mean_rating"], int(row["image_id"])))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(10)]
    for index, row in enumerate(ordered):
        buckets[min(9, index * 10 // len(ordered))].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    base, remainder = divmod(limit, 10)
    for index, bucket in enumerate(buckets):
        rng.shuffle(bucket)
        # Allocate the remainder one image at a time so arbitrary smoke sizes
        # stay balanced instead of truncating the lowest-score buckets.
        quota = base + (1 if index < remainder else 0)
        selected.extend(bucket[:quota])
    selected.sort(key=lambda row: int(row["image_id"]))
    return selected


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for position, row in enumerate(rows):
            payload = dict(row)
            payload["manifest_position"] = position
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="deterministic stratified smoke size")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    rows, rejects = load_rows(args.annotations, args.image_dir)
    if not rows:
        raise SystemExit("no usable AVA rows found")
    selected = stratified_sample(rows, args.limit, args.seed) if args.limit else rows
    write_jsonl(selected, args.output)
    audit = {
        "annotation_path": str(args.annotations.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "usable_annotations": len(rows),
        "selected_rows": len(selected),
        "rejects": rejects,
        "seed": args.seed,
        "mean_rating_range": [min(row["mean_rating"] for row in selected), max(row["mean_rating"] for row in selected)],
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
