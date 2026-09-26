"""Prepare TAD66K or SPAQ into canonical JSONL manifests; never downloads data."""
from __future__ import annotations
import argparse
from pathlib import Path
from image_aesthetics.data.adapters import KonIQ10KAdapter, SPAQAdapter, TAD66KAdapter
from image_aesthetics.data.preparation import prepare_dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("tad66k", "spaq", "koniq10k"))
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    adapter = (TAD66KAdapter(args.annotations) if args.dataset == "tad66k" else
               SPAQAdapter(args.annotations) if args.dataset == "spaq" else
               KonIQ10KAdapter(args.annotations))
    print(prepare_dataset(adapter, image_root=args.image_root, output_root=args.output_root, source_files=(args.annotations,)))

if __name__ == "__main__": main()
