#!/usr/bin/env python3
"""Evaluate the pretrained PyIQA NIMA model on an immutable AVA split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AVASplit(Dataset):
    def __init__(self, manifest: Path, dataset_manifest: Path, limit: int | None) -> None:
        sources = {
            row["image_id"]: row["source_path"]
            for row in (json.loads(line) for line in dataset_manifest.read_text().splitlines() if line.strip())
        }
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        for row in rows:
            row["source_path"] = sources[row["image_id"]]
        self.rows = rows[:limit] if limit else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["source_path"]) if "source_path" in row else None
        if path is None:
            raise KeyError("split row requires source_path")
        with Image.open(path) as image:
            image.load()
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = pil_to_tensor(image).float().div_(255.0)
        votes = torch.tensor(row["vote_counts"], dtype=torch.float32)
        votes /= votes.sum()
        return row["image_id"], tensor, float(row["mean_rating"]), votes


def collate(batch):
    # NIMA performs its own resize and center crop, but tensors in a batch must
    # first share dimensions. Applying the documented evaluation transform here
    # preserves aspect ratio and produces the model's 299px input.
    from torchvision.transforms.functional import center_crop, resize

    images, ids, means, votes = [], [], [], []
    for image_id, tensor, mean, distribution in batch:
        tensor = resize(tensor, 299, antialias=True)
        tensor = center_crop(tensor, [299, 299])
        images.append(tensor)
        ids.append(image_id)
        means.append(mean)
        votes.append(distribution)
    return ids, torch.stack(images), np.asarray(means), torch.stack(votes)


def pairwise_accuracy(predicted: np.ndarray, target: np.ndarray, seed: int = 20260929) -> float:
    rng = random.Random(seed)
    correct = total = 0
    count = len(predicted)
    for _ in range(min(200_000, count * 20)):
        a, b = rng.randrange(count), rng.randrange(count)
        if a == b or target[a] == target[b]:
            continue
        correct += int((predicted[a] > predicted[b]) == (target[a] > target[b]))
        total += 1
    return correct / total if total else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "status.json"
    predictions_path = args.output / "predictions.jsonl"
    report_path = args.output / "report.json"
    dataset = AVASplit(args.manifest, args.dataset_manifest, args.limit)

    completed: dict[str, dict] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["image_id"]] = row

    import pyiqa

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    metric = pyiqa.create_metric("nima", device=device)
    metric.net.eval()
    start = time.time()
    processed = 0
    atomic_json(status_path, {"state": "running", "completed": len(completed), "total": len(dataset), "device": device})

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate, persistent_workers=args.workers > 0)
    with predictions_path.open("a", buffering=1) as sink, torch.inference_mode():
        for ids, images, targets, target_dist in loader:
            missing = [index for index, image_id in enumerate(ids) if image_id not in completed]
            if not missing:
                continue
            selected = images[missing].to(device)
            mos, distribution = metric.net(selected, return_mos=True, return_dist=True)
            mos = mos.detach().cpu().numpy().reshape(-1)
            distribution = distribution.detach().cpu().numpy()
            for local, source_index in enumerate(missing):
                row = {
                    "image_id": ids[source_index],
                    "target_mean": float(targets[source_index]),
                    "predicted_mean": float(mos[local]),
                    "target_distribution": target_dist[source_index].tolist(),
                    "predicted_distribution": distribution[local].tolist(),
                }
                sink.write(json.dumps(row, sort_keys=True) + "\n")
                completed[row["image_id"]] = row
            processed += len(missing)
            if processed % 100 <= len(missing):
                elapsed = time.time() - start
                atomic_json(status_path, {"state": "running", "completed": len(completed), "total": len(dataset),
                                         "device": device, "images_per_second": processed / elapsed,
                                         "elapsed_seconds": elapsed})

    ordered = [completed[row["image_id"]] for row in dataset.rows]
    predicted = np.asarray([row["predicted_mean"] for row in ordered])
    target = np.asarray([row["target_mean"] for row in ordered])
    pred_dist = np.asarray([row["predicted_distribution"] for row in ordered])
    target_dist = np.asarray([row["target_distribution"] for row in ordered])
    cdf_delta = np.cumsum(pred_dist, axis=1) - np.cumsum(target_dist, axis=1)
    report = {
        "state": "completed",
        "rows": len(ordered),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": device,
        "pearson": float(pearsonr(predicted, target).statistic),
        "spearman": float(spearmanr(predicted, target).statistic),
        "binary_accuracy_mean_5": float(np.mean((predicted >= 5.0) == (target >= 5.0))),
        "pairwise_accuracy": pairwise_accuracy(predicted, target),
        "mae": float(np.mean(np.abs(predicted - target))),
        "rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "emd": float(np.mean(np.mean(np.abs(cdf_delta), axis=1))),
        "predicted_mean": float(np.mean(predicted)),
        "predicted_std": float(np.std(predicted)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "elapsed_seconds": time.time() - start,
    }
    atomic_json(report_path, report)
    atomic_json(status_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
