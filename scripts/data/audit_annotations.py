#!/usr/bin/env python3
"""Audit native annotations without extracting or decoding dataset images."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.adapters import SPAQAdapter, TAD66KAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("tad66k", "spaq"))
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    adapter = (
        TAD66KAdapter(args.annotations)
        if args.dataset == "tad66k"
        else SPAQAdapter(args.annotations)
    )
    payload = asdict(adapter.inventory())
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    print(encoded, end="")
    return 0 if payload["rejected"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
