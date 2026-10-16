#!/usr/bin/env python3
"""Small deterministic CPU/MPS smoke run for F3C and the F5 candidates.

This uses synthetic in-memory development rows only. It does not open any data
manifest, cache, external dataset, or locked-test artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from image_aesthetics.models.fusion import RegularizedLinear, build_baseline
from image_aesthetics.training.fusion_trainer import train_fusion


class SyntheticFusionRows(Dataset):
    def __init__(self, count: int, *, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.features = torch.randn(count, 1544, generator=generator)
        self.targets = torch.sigmoid(self.features[:, 1540] * 0.2 + self.features[:, 768] * 0.1)
        self.samples = [
            ("synthetic", index,
             {"native_score_range": [0.0, 1.0], "group_id": f"g{index}"},
             float(self.targets[index]))
            for index in range(count)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return {
            "features": self.features[index],
            "target": self.targets[index],
            "native_target": self.targets[index],
            "dataset_id": "synthetic",
            "image_id": str(index),
            "group_id": f"g{index}",
            "pairing_group": "synthetic",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 8 or args.epochs < 1:
        parser.error("--samples must be >= 8 and --epochs must be >= 1")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable")

    train_data = SyntheticFusionRows(args.samples, seed=51)
    validation_data = SyntheticFusionRows(max(8, args.samples // 2), seed=52)
    results = {}
    for index, name in enumerate(("F3C", "F5-R", "F5-G", "F5-X")):
        torch.manual_seed(700 + index)
        model = build_baseline(name)
        initialization = None
        if name.startswith("F5-"):
            torch.manual_seed(800 + index)
            source_f2 = RegularizedLinear().to(args.device)
            model.initialize_from_f2(source_f2.state_dict())
            before = model(train_data.features.to(args.device))
            expected = source_f2(train_data.features.to(args.device))
            torch.testing.assert_close(before, expected, rtol=0, atol=0)
            initialization = "synthetic-fold-f2"
        with tempfile.TemporaryDirectory(prefix=f"fusion-{name}-smoke-") as output:
            config = {
                "identity": f"synthetic-{name}-v1",
                "model": name,
                "model_config": (model.architecture_config()
                                 if hasattr(model, "architecture_config")
                                 else {"model_id": "F3C-no-per-image-layernorm-v1"}),
                "f2_initialization": initialization,
                "locked_test_accessed": False,
            }
            status = train_fusion(
                model, train_data, validation_data, Path(output), config=config,
                epochs=args.epochs, min_epochs=args.epochs, patience=1,
                batch_size=8, seed=900 + index, device=args.device,
                ranking_weight=0, status_interval_batches=1, num_workers=0,
            )
            assert status["state"] == "completed"
            assert (Path(output) / "checkpoints" / "best.pt").is_file()
            results[name] = {
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "epoch": status["epoch"],
                "best_epoch": status["best_epoch"],
                "best_macro_spearman": status["best_macro_spearman"],
            }
    print(json.dumps({"device": args.device, "samples": args.samples,
                      "epochs": args.epochs, "results": results}, indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
