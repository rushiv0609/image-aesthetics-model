#!/usr/bin/env python3
"""Inspect the dataset registry; never downloads or writes external data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from image_aesthetics.data.registry import acquisition_plan, load_registry, validate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "show", "validate", "acquisition-plan"))
    parser.add_argument("dataset_id", nargs="?")
    parser.add_argument("--registry-dir", type=Path, default=ROOT / "configs/data/datasets")
    parser.add_argument(
        "--use-mode", choices=("commercial-safe", "non-commercial-research"),
        default="commercial-safe",
    )
    parser.add_argument("--show-excluded", action="store_true")
    args = parser.parse_args()
    records = load_registry(args.registry_dir)
    errors = validate(records)
    if args.command == "validate":
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"valid: {len(records)} datasets")
        return 0
    if errors:
        print("registry invalid; run validate first", file=sys.stderr)
        return 1
    if args.command == "list":
        for record in records:
            print(f"{record.dataset_id}\t{record.task}\t{'enabled' if record.enabled else 'disabled'}")
    elif args.command == "show":
        matches = [r for r in records if r.dataset_id == args.dataset_id]
        if not matches:
            print(f"unknown dataset: {args.dataset_id}", file=sys.stderr)
            return 2
        print(json.dumps(matches[0].__dict__ | {"path": str(matches[0].path) if matches[0].path else None}, indent=2, default=list, sort_keys=True))
    else:
        included = acquisition_plan(records, args.use_mode)
        for record in included:
            print(f"{record.dataset_id}\t{record.download_method}\t{record.official_url}")
        if args.show_excluded:
            included_ids = {record.dataset_id for record in included}
            for record in records:
                if record.dataset_id not in included_ids:
                    _, reason = record.acquisition_decision(args.use_mode)
                    print(f"EXCLUDED\t{record.dataset_id}\t{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
