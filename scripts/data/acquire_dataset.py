#!/usr/bin/env python3
"""Acquire one registered dataset; manual/form sources only print instructions."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from image_aesthetics.data.acquisition import TOOL_VERSION, download, safe_member_path, write_manifest
from image_aesthetics.data.registry import load_registry

def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--dataset", required=True); p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--url"); p.add_argument("--filename", default="source.download"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--user-supplied", action="store_true"); args = p.parse_args()
    records = {r.dataset_id: r for r in load_registry(ROOT / "configs/data/datasets")}
    if args.dataset not in records: p.error("unknown dataset")
    record = records[args.dataset]; allowed, reason = record.acquisition_decision("non-commercial-research")
    if not allowed: print(f"blocked: {reason}", file=sys.stderr); return 2
    root = args.data_root / args.dataset
    if record.download_method != "direct":
        supplied = safe_member_path(root, args.filename)
        if args.user_supplied and supplied.is_file():
            write_manifest(root, url=record.official_url, source="user-supplied", path=supplied, user_supplied=True)
            print(f"recorded user-supplied file manifest: {root / args.filename}")
            return 0
        print(f"Manual access required for {record.display_name}. Use the official source: {record.official_url}")
        print("After downloading through the provider, rerun with --user-supplied and --filename <file> to record its manifest.")
        return 0
    if not args.url: p.error("--url is required for direct downloads")
    try: destination = safe_member_path(root, args.filename)
    except ValueError as exc: p.error(str(exc))
    if args.dry_run: print(f"would download {args.url} -> {destination} ({TOOL_VERSION})"); return 0
    download(args.url, destination); write_manifest(root, url=args.url, source="direct", path=destination, user_supplied=args.user_supplied)
    return 0
if __name__ == "__main__": raise SystemExit(main())
