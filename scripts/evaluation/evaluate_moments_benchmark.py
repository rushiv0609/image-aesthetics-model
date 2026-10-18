#!/usr/bin/env python3
"""Evaluate fixed scorer predictions on a blinded Moments benchmark."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from image_aesthetics.evaluation.moments_benchmark import evaluate_moments_predictions


def _jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_moments_predictions(_jsonl(args.manifest), _jsonl(args.annotations),
                                          _jsonl(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
