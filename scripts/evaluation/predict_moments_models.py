#!/usr/bin/env python3
"""Generate provenance-rich F0/F1/F4/F5-G scores for a joined-ten benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.joined_ten_dataset import JoinedTenDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import MOMENTS_F0_WEIGHTS, build_baseline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(name: str, checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_baseline(name)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def _isotonic(values: np.ndarray, artifact: Path) -> np.ndarray:
    data = json.loads(artifact.read_text())
    if data.get("schema_version") != "shared-isotonic-v1" or data.get("locked_test_accessed") is not False:
        raise ValueError("invalid shared isotonic artifact")
    x = np.asarray(data["thresholds"], dtype=np.float64)
    y = np.asarray(data["values"], dtype=np.float64)
    if x.ndim != 1 or len(x) != len(y) or len(x) < 2 or np.any(np.diff(x) < 0):
        raise ValueError("invalid isotonic knots")
    return np.interp(values, x, y, left=y[0], right=y[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--join-dir", type=Path, required=True)
    parser.add_argument("--f4-root", type=Path, required=True)
    parser.add_argument("--f5-root", type=Path, required=True)
    parser.add_argument("--normalization-root", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    dataset = JoinedTenDataset(args.join_dir)
    def collate(rows):
        return {"features": torch.stack([row["features"] for row in rows]),
                "image_id": [row["image_id"] for row in rows]}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate)
    item_ids: list[str] = []
    batches: list[torch.Tensor] = []
    for batch in loader:
        item_ids.extend(map(str, batch["image_id"]))
        batches.append(batch["features"].float())
    if len(item_ids) != len(dataset) or len(set(item_ids)) != len(item_ids):
        raise ValueError("joined benchmark IDs are incomplete or duplicated")

    predictions: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        f0 = build_baseline("F0", weights=MOMENTS_F0_WEIGHTS).eval()
        f1 = build_baseline("F1", mode="mean").eval()
        predictions["F0"] = np.concatenate([f0(x).numpy() for x in batches])
        predictions["F1-mean"] = np.concatenate([f1(x).numpy() for x in batches])

    provenance: dict[str, list[dict[str, object]]] = {"F4": [], "F5-G": []}
    for model_name, root in (("F4", args.f4_root), ("F5-G", args.f5_root)):
        members = []
        for fold in range(5):
            normalizer_path = args.normalization_root / f"fold{fold}.json"
            normalizer = FeatureNormalizer.from_file(normalizer_path)
            for seed in (20260920, 20260921, 20260922):
                checkpoint = root / f"{model_name}-fold{fold}-seed{seed}" / "checkpoints" / "best.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                model = _load_model(model_name, checkpoint)
                with torch.inference_mode():
                    member = np.concatenate([model(normalizer.transform(x)).numpy() for x in batches])
                members.append(member)
                provenance[model_name].append({"fold": fold, "seed": seed,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
                    "normalizer": str(normalizer_path), "normalizer_sha256": _sha256(normalizer_path)})
        predictions[model_name] = np.mean(np.stack(members), axis=0)
    predictions["F5-G-shared-isotonic"] = _isotonic(predictions["F5-G"], args.calibrator)

    for name, values in predictions.items():
        if values.shape != (len(item_ids),) or not np.isfinite(values).all():
            raise ValueError(f"invalid prediction vector for {name}")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"predictions outside [0,1] for {name}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "predictions.jsonl"
    with output.open("w") as handle:
        for model_name, values in predictions.items():
            for item_id, score in zip(item_ids, values, strict=True):
                handle.write(json.dumps({"item_id": item_id, "model": model_name,
                                         "score": float(score)}, sort_keys=True) + "\n")
    audit = json.loads((args.join_dir / "join-audit.json").read_text())
    report = {"schema_version": "moments-model-predictions-v1", "locked_test_accessed": False,
              "human_labels_accessed": False, "item_count": len(item_ids),
              "models": {name: {"min": float(v.min()), "max": float(v.max()),
                                  "mean": float(v.mean())} for name, v in predictions.items()},
              "join_audit_sha256": _sha256(args.join_dir / "join-audit.json"),
              "join_identity": audit.get("identity"), "calibrator": str(args.calibrator),
              "calibrator_sha256": _sha256(args.calibrator), "members": provenance,
              "predictions_sha256": _sha256(output)}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
