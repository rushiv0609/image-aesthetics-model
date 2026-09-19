#!/usr/bin/env python3
"""Fit the locked A1 configuration on development data and evaluate test once."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from image_aesthetics.data.training_dataset import CachedAVADataset
from image_aesthetics.evaluation.metrics import metric_report
from image_aesthetics.models.siglip_head.losses import distribution_loss
from image_aesthetics.models.siglip_head.models import build_head


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def evaluate(model, ds, device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    pred, target = [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            pred.append(model(x.to(device)).softmax(-1).cpu().numpy())
            target.append(y.numpy())
    return metric_report(np.concatenate(pred), np.concatenate(target))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/Users/rushivyas/datasets/image-aesthetics/ava")
    p.add_argument("--output-root", default=None)
    p.add_argument("--epochs", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=20260920)
    args = p.parse_args()
    if args.epochs != 9:
        raise ValueError("A1 finalization is locked to the development-CV median of 9 epochs")

    root = Path(args.data_root)
    split = root / "prepared/v1/splits"
    cache = root / "embeddings/siglip2-base-patch16-224/full-v1"
    split_meta = json.loads((split / "split-metadata.json").read_text())
    cache_meta = json.loads((cache / "manifest.json").read_text())
    output = Path(args.output_root or root / "releases/siglip2-aesthetic/a1-v1")
    if output.exists():
        raise FileExistsError(
            f"immutable final release already exists: {output}; "
            "refusing to refit or reopen the locked test"
        )
    output.mkdir(parents=True, exist_ok=False)

    train_ds = CachedAVADataset(split / "development.jsonl", cache, max_shards=64)
    model = build_head("A1")
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    model.to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    total = args.epochs * len(loader); warm = max(1, int(.05 * total))
    def lrfn(step):
        return min(1., (step + 1) / warm) * (.5 * (1 + np.cos(np.pi * max(0, step - warm) / max(1, total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrfn)
    history = []
    for epoch in range(args.epochs):
        model.train(); losses = []; emds = []; hubs = []
        for x, y, _ in loader:
            loss, parts = distribution_loss(model(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            losses.append(float(loss.detach())); emds.append(parts["emd"]); hubs.append(parts["huber"])
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "emd": float(np.mean(emds)), "huber": float(np.mean(hubs)),
                        "lr": opt.param_groups[0]["lr"]})

    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    model_path = output / "model.safetensors"
    save_file(state, str(model_path))
    # Deliberately materialize the locked-test dataset only after fitting/export.
    test_ds = CachedAVADataset(split / "locked-test.jsonl", cache, max_shards=64, allow_locked_test=True)
    test_metrics = evaluate(model, test_ds, device, args.batch_size)
    config = {
        "architecture": "A1", "input_dim": 768, "output_dim": 10,
        "layers": [768, 256, 10], "dropout": [0.10], "param_count": model.param_count,
        "loss": "squared_emd+0.25_huber", "optimizer": "AdamW",
        "learning_rate": 3e-4, "weight_decay": 1e-4, "batch_size": args.batch_size,
        "epochs": args.epochs, "seed": args.seed, "device": str(device),
        "selection_policy": "9 epochs frozen before locked-test access; median A1 best epoch across 15 development-only CV runs",
    }
    preprocessing = {"embedding": "google/siglip2-base-patch16-224", "revision": split_meta["model_revision"],
                     "dimension": 768, "l2_normalized": True, "head_output": "expected_rating_normalized_0_1",
                     "score": "(sum(softmax(logits)[i] * (i+1)) - 1) / 9"}
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                 "torch_version": torch.__version__, "split_manifest_sha256": sha256(split / "split-metadata.json"),
                 "development_sha256": split_meta["outer"]["development"]["sha256"],
                 "locked_test_sha256": split_meta["outer"]["test"]["sha256"],
                 "cache_manifest_sha256": cache_meta.get("manifest_sha256", split_meta["cache_manifest_sha256"]),
                 "test_access": "single final evaluation after fit; no test metrics used for selection"}
    (output / "model-config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2, sort_keys=True) + "\n")
    (output / "metrics.json").write_text(json.dumps({"locked_test": test_metrics, "fit_history": history}, indent=2, sort_keys=True) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"release": str(output), "test": test_metrics, "model_sha256": sha256(model_path)}, indent=2))


if __name__ == "__main__":
    main()
