#!/usr/bin/env python3
"""Freeze a validated Moments manifest after label-blind decode preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


ALLOWED_ERRORS = {"UnidentifiedImageError", "SourceDecodeError"}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                            for row in rows))


def filter_unreadable(benchmark_manifest: Path, feature_manifest: Path,
                      annotations: Path, failures: Path | None, output_dir: Path,
                      *, failure_rows: list[dict] | None = None) -> dict:
    if Path(annotations).read_text().strip():
        raise ValueError("cannot change benchmark membership after human annotation")
    benchmark = _rows(benchmark_manifest); features = _rows(feature_manifest)
    if failure_rows is None:
        if failures is None:
            raise ValueError("failure rows or a failure report are required")
        failure_rows = json.loads(Path(failures).read_text())
    if not isinstance(failure_rows, list) or not failure_rows:
        raise ValueError("a non-empty cache failure report is required")
    if any(row.get("error_type") not in ALLOWED_ERRORS for row in failure_rows):
        raise ValueError("only deterministic media decode failures may be excluded")
    failed = {str(row["image_id"]) for row in failure_rows}
    benchmark_ids = {str(row["item_id"]) for row in benchmark}
    feature_ids = {str(row["image_id"]) for row in features}
    if len(benchmark_ids) != len(benchmark) or len(feature_ids) != len(features):
        raise ValueError("source manifests contain duplicate IDs")
    if benchmark_ids != feature_ids or not failed <= benchmark_ids:
        raise ValueError("benchmark, feature, and failure identities do not align")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"validated output must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_benchmark = [row for row in benchmark if str(row["item_id"]) not in failed]
    valid_features = [row for row in features if str(row["image_id"]) not in failed]
    exclusions = [{"item_id": str(row["image_id"]), "error_type": row["error_type"],
                   "error_message": str(row.get("error_message", ""))}
                  for row in failure_rows]
    benchmark_out = output_dir / "manifest.private.validated.jsonl"
    features_out = output_dir / "features.validated.jsonl"
    exclusions_out = output_dir / "decode-exclusions.private.jsonl"
    _write_jsonl(benchmark_out, valid_benchmark)
    _write_jsonl(features_out, valid_features)
    _write_jsonl(exclusions_out, exclusions)
    report = {
        "schema_version": "moments-decode-validation-v1", "state": "completed",
        "source_items": len(benchmark), "valid_items": len(valid_benchmark),
        "excluded_items": len(failed), "human_annotations_present": False,
        "locked_test_accessed": False,
        "source_benchmark_sha256": _sha(benchmark_manifest),
        "source_feature_manifest_sha256": _sha(feature_manifest),
        "cache_failures_sha256": hashlib.sha256(json.dumps(
            failure_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "validated_benchmark_sha256": _sha(benchmark_out),
        "validated_feature_manifest_sha256": _sha(features_out),
    }
    (output_dir / "validation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--failures", type=Path)
    source.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    cache_dir = args.cache_dir; del args.cache_dir
    failure_rows = None
    if cache_dir is not None:
        database = Path(cache_dir) / "work.sqlite3"
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            columns = ("image_id", "state", "error_type", "error_message")
            failure_rows = [dict(zip(columns, row)) for row in connection.execute(
                "SELECT image_id,state,error_type,error_message FROM items "
                "WHERE state LIKE '%error' ORDER BY image_id")]
    print(json.dumps(filter_unreadable(**vars(args), failure_rows=failure_rows),
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
