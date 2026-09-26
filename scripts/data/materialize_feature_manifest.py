"""Normalize an existing prepared dataset snapshot for feature extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from image_aesthetics.data.manifest_materialization import DATASETS, materialize_feature_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_feature_manifest(
        args.dataset, args.prepared_root, args.image_root, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
