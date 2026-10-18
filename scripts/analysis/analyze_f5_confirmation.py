#!/usr/bin/env python3
"""Read-only diagnostics for completed F5 confirmation validation runs.

The script deliberately excludes locked-test rows before any analysis and only
loads validation partitions for the selected development fold.  It emits
versioned JSON and Markdown reports; it never updates checkpoints or training
state.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.models.fusion import build_baseline

DATASETS = ("aadb", "flickr_aes", "para", "tad66k")
SEEDS = (20260920, 20260921, 20260922)
VERSION = "f5-confirmation-diagnostics-v1"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        indexes = np.flatnonzero(values == value)
        if len(indexes) > 1:
            result[indexes] = result[indexes].mean()
    return result


def _spearman(prediction, target) -> float:
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    if len(p) < 2 or p.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(_rank(p), _rank(y))[0, 1])


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    p = np.asarray([row["prediction"] for row in rows])
    y = np.asarray([row["target"] for row in rows])
    r = np.asarray([row["residual"] for row in rows])
    base = np.asarray([row["base_prediction"] for row in rows])
    result = {
        "n": len(rows), "spearman": _spearman(p, y),
        "base_spearman": _spearman(base, y),
        "spearman_delta_vs_internal_base": _spearman(p, y) - _spearman(base, y),
        "mae": float(np.abs(p - y).mean()),
        "base_mae": float(np.abs(base - y).mean()),
        "mean_residual": float(r.mean()), "mean_abs_residual": float(np.abs(r).mean()),
        "residual_positive_fraction": float((r > 0).mean()),
        "residual_near_bound_fraction": float((np.abs(r) >= .225).mean()),
        "prediction_mean": float(p.mean()), "target_mean": float(y.mean()),
    }
    gates = [v for row in rows for v in row.get("gates", [])]
    if gates:
        gates = np.asarray(gates)
        result.update({"gate_mean": float(gates.mean()),
                       "gate_mean_abs_deviation_from_one": float(np.abs(gates - 1).mean()),
                       "gate_near_bound_fraction": float((np.abs(gates - 1) >= .225).mean())})
    return result


def _quantile_groups(rows: list[dict], key: str, bins: int = 10) -> list[dict]:
    if not rows:
        return []
    values = np.asarray([row[key] for row in rows], float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return [{"bin": 0, "low": float(values[0]), "high": float(values[0]), **_summary(rows)}]
    assignments = np.searchsorted(edges[1:-1], values, side="right")
    return [{"bin": int(index), "low": float(values[assignments == index].min()),
             "high": float(values[assignments == index].max()),
             **_summary([row for row, assigned in zip(rows, assignments) if assigned == index])}
            for index in range(len(edges) - 1) if np.any(assignments == index)]


def _split_audit(manifest: Path) -> dict:
    # Parse only development rows. Locked rows are discarded before their
    # identifiers, labels, or metadata are added to any analysis structure.
    development = []
    locked_excluded = 0
    with manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("locked_test"):
                locked_excluded += 1
                continue
            development.append(row)
    result = {"development_rows": len(development), "locked_rows_excluded": locked_excluded,
              "locked_test_accessed": False, "duplicate_record_ids": 0,
              "group_cross_fold_overlap": [], "source_group_cross_fold_overlap": []}
    identifiers = collections.Counter(str(row["record_id"]) for row in development)
    result["duplicate_record_ids"] = sum(count - 1 for count in identifiers.values() if count > 1)
    for field, output in (("group_id", "group_cross_fold_overlap"),
                          ("source_groups", "source_group_cross_fold_overlap")):
        seen = collections.defaultdict(set)
        for row in development:
            values = row.get(field) or []
            if not isinstance(values, list):
                values = [values]
            for value in values:
                seen[f"{row['dataset_id']}:{value}"].add(int(row["development_fold"]))
        result[output] = sorted(key for key, folds in seen.items() if len(folds) > 1)[:100]
        result[output + "_count"] = sum(len(folds) > 1 for folds in seen.values())
    result["fold_dataset_counts"] = {
        str(fold): dict(collections.Counter(row["dataset_id"] for row in development
                                            if int(row["development_fold"]) == fold))
        for fold in range(5)}
    return result


def _load_validation(data_root: Path, baseline_root: Path, fold: int):
    manifest = data_root / "fusion/prepared/v6/samples.jsonl"
    joins = {name: data_root / name / "features/joined-ten-v1" for name in DATASETS}
    base = FusionTrainingDataset(manifest, joins, partition="validation", validation_fold=fold)
    train = FusionTrainingDataset(manifest, joins, partition="train", validation_fold=fold)
    normalizer = FeatureNormalizer.from_file(
        baseline_root / "normalization" / f"fold{fold}.json",
        expected_identity=train.normalization_identity())
    return base, base.normalized(normalizer)


def _infer_run(run_dir: Path, normalized, raw_rows, *, expected_fold: int,
               expected_seed: int, expected_identity: str) -> tuple[list[dict], dict]:
    config = json.loads((run_dir / "config.json").read_text())
    status_path = run_dir / "training-status.json"
    metadata_path = run_dir / "checkpoints/checkpoint.json"
    status = json.loads(status_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    identity = config.get("identity")
    if (config.get("run_id") != run_dir.name or config.get("fold") != expected_fold
            or config.get("seed") != expected_seed or config.get("model") != "F5-G"
            or identity != expected_identity or config.get("locked_test_accessed") is not False
            or status.get("state") != "completed" or status.get("identity") != identity
            or status.get("locked_test_accessed") is not False
            or metadata.get("identity") != identity):
        raise ValueError(f"unsafe or unexpected run config: {run_dir}")
    checkpoint_path = run_dir / "checkpoints/best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_baseline("F5-G")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = []
    offset = 0
    with torch.no_grad():
        for batch in DataLoader(normalized, batch_size=1024, shuffle=False, num_workers=0):
            x = batch["features"].float()
            prediction = model(x)
            residual = model.bounded_residual(x)
            _, _, scalars = __import__("image_aesthetics.models.fusion", fromlist=["split_features"]).split_features(x)
            base = torch.sigmoid(model.scalar_base(scalars).squeeze(-1))
            gates = model.gate_values(x)
            count = len(prediction)
            for index in range(count):
                source = raw_rows.samples[offset + index][2]
                output.append({
                    "fold": int(config["fold"]), "seed": int(config["seed"]),
                    "dataset": str(batch["dataset_id"][index]),
                    "image_id": str(batch["image_id"][index]),
                    "group_id": str(batch["group_id"][index]),
                    "target": float(batch["target"][index]),
                    "prediction": float(prediction[index]),
                    "base_prediction": float(base[index]),
                    "residual": float(residual[index]),
                    "gates": gates[index].tolist(),
                    "themes": list(source.get("themes") or []),
                })
            offset += count
    provenance = {
        "identity": identity,
        "config_sha256": hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest(),
        "status_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        "checkpoint_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "best_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    return output, provenance


def _markdown(report: dict) -> str:
    lines = ["# F5 confirmation diagnostic report", "", f"Version: `{VERSION}`", "",
             "Locked test was not accessed. All prediction diagnostics use development validation folds only.", "",
             "## Dataset summary", "", "| Dataset | N predictions | Spearman | Internal F2-base Spearman | Delta | MAE | Mean |residual| | Near bound |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["by_dataset"].items():
        lines.append(f"| {name} | {row['n']} | {row['spearman']:.4f} | {row['base_spearman']:.4f} | {row['spearman_delta_vs_internal_base']:+.4f} | {row['mae']:.4f} | {row['mean_abs_residual']:.4f} | {row['residual_near_bound_fraction']:.1%} |")
    tad = report["by_dataset"]["tad66k"]
    lines += ["", "## TAD66K diagnosis", "",
              f"Across all fold/seed validation predictions, F5-G changes the internal scalar base Spearman by **{tad['spearman_delta_vs_internal_base']:+.4f}**. Mean absolute residual is **{tad['mean_abs_residual']:.4f}** and **{tad['residual_near_bound_fraction']:.1%}** of residuals are within 10% of the configured bound.", "",
              "Worst themes below require at least 100 validation predictions (pooled across seeds):", "",
              "| Theme | N | F5 Spearman | Base Spearman | Delta |", "|---|---:|---:|---:|---:|"]
    for name, row in report["tad66k_themes_worst_delta"][:15]:
        lines.append(f"| {name} | {row['n']} | {row['spearman']:.4f} | {row['base_spearman']:.4f} | {row['spearman_delta_vs_internal_base']:+.4f} |")
    audit = report["split_audit"]
    lines += ["", "## Split-integrity audit", "",
              f"Development rows: {audit['development_rows']:,}; locked rows excluded before analysis: {audit['locked_rows_excluded']:,}.",
              f"Duplicate record IDs: {audit['duplicate_record_ids']}; cross-fold group overlaps: {audit['group_cross_fold_overlap_count']}; cross-fold source-group overlaps: {audit['source_group_cross_fold_overlap_count']}.", "",
              "## Limitations", "", "The prepared manifest has TAD66K theme tags but no richer semantic category or technical-regime annotations. This report does not infer unrecorded themes from embeddings or images. Repeated seed predictions are correlated and must not be treated as independent images.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--baseline-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/baseline-matrix-v6"))
    parser.add_argument("--confirmation-root", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/f5-confirmation-v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("/Users/rushivyas/datasets/image-aesthetics/fusion/experiments/f5-confirmation-diagnostics-v1"))
    args = parser.parse_args()
    aggregate_path = args.confirmation_root / "aggregate-metrics.json"
    aggregate = json.loads(aggregate_path.read_text())
    if (aggregate.get("experiment_version") != "f5-confirmation-v1"
            or aggregate.get("state") != "completed" or aggregate.get("failed")
            or aggregate.get("pending") or aggregate.get("locked_test_accessed") is not False
            or not isinstance(aggregate.get("summaries"), dict)):
        raise ValueError(f"confirmation aggregate is incomplete or unsafe: {aggregate_path}")
    all_rows = []
    per_run = {}
    source_runs = {}
    for fold in range(5):
        raw, normalized = _load_validation(args.data_root, args.baseline_root, fold)
        for seed in SEEDS:
            run_id = f"F5-G-fold{fold}-seed{seed}"
            summary = aggregate["summaries"].get(run_id)
            if not isinstance(summary, dict) or not summary.get("identity"):
                raise ValueError(f"confirmation aggregate missing source run: {run_id}")
            rows, source_runs[run_id] = _infer_run(
                args.confirmation_root / run_id, normalized, raw,
                expected_fold=fold, expected_seed=seed,
                expected_identity=summary["identity"])
            per_run[run_id] = _summary(rows)
            all_rows.extend(rows)
    by_dataset = {name: _summary([row for row in all_rows if row["dataset"] == name]) for name in DATASETS}
    tad = [row for row in all_rows if row["dataset"] == "tad66k"]
    themes = collections.defaultdict(list)
    for row in tad:
        for theme in row["themes"]:
            themes[str(theme)].append(row)
    theme_summary = [(name, _summary(rows)) for name, rows in themes.items() if len(rows) >= 100]
    theme_summary.sort(key=lambda item: item[1]["spearman_delta_vs_internal_base"])
    report = {
        "version": VERSION, "locked_test_accessed": False,
        "inputs": {"confirmation_root": str(args.confirmation_root),
                   "confirmation_aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
                   "source_runs": source_runs},
        "prediction_rows": len(all_rows), "unique_validation_images": len({row["image_id"] for row in all_rows}),
        "by_run": per_run, "by_dataset": by_dataset,
        "by_fold": {str(fold): _summary([row for row in all_rows if row["fold"] == fold])
                    for fold in range(5)},
        "by_label_decile": {name: _quantile_groups([row for row in all_rows if row["dataset"] == name], "target") for name in DATASETS},
        "by_prediction_decile": {name: _quantile_groups([row for row in all_rows if row["dataset"] == name], "prediction") for name in DATASETS},
        "by_residual_magnitude_decile": {name: _quantile_groups([{**row, "abs_residual": abs(row["residual"])} for row in all_rows if row["dataset"] == name], "abs_residual") for name in DATASETS},
        "by_residual_sign": {name: {
            sign: _summary([row for row in all_rows if row["dataset"] == name and
                            ((row["residual"] >= 0) if sign == "positive" else (row["residual"] < 0))])
            for sign in ("positive", "negative")}
            for name in DATASETS},
        "tad66k_themes_worst_delta": theme_summary,
        "split_audit": _split_audit(args.data_root / "fusion/prepared/v6/samples.jsonl"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "report.md").write_text(_markdown(report))
    print(json.dumps({"output_dir": str(args.output_dir), "prediction_rows": len(all_rows),
                      "unique_validation_images": report["unique_validation_images"],
                      "locked_test_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
