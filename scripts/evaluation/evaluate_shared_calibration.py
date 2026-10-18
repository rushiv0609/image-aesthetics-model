#!/usr/bin/env python3
"""Cross-fit and export one shared monotonic calibrator from OOF predictions.

Input is JSONL containing development-only, out-of-fold predictions.  The
workflow never accepts a locked-test override and never uses dataset identity
as a calibrator input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from image_aesthetics.evaluation.monotonic_calibration import fit_isotonic, validate_oof_rows
from image_aesthetics.evaluation.metrics import _rank, corr


def _metrics(prediction, target):
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    spearman = corr(_rank(p), _rank(y))
    return {"n": len(p), "mae": float(np.mean(np.abs(p - y))),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))), "spearman": spearman}


def evaluate(rows):
    rows = validate_oof_rows(rows)
    folds = sorted({int(row["development_fold"]) for row in rows})
    if len(folds) < 2:
        raise ValueError("cross-fitted calibration requires at least two development folds")
    outputs = []
    fold_reports = {}
    for fold in folds:
        fit = [row for row in rows if int(row["development_fold"]) != fold]
        held = [row for row in rows if int(row["development_fold"]) == fold]
        fit_groups = {(str(row["dataset_id"]), str(row["group_id"])) for row in fit}
        held_groups = {(str(row["dataset_id"]), str(row["group_id"])) for row in held}
        if fit_groups & held_groups:
            raise ValueError(f"group leakage across calibration fold {fold}")
        calibrator = fit_isotonic([float(row["prediction"]) for row in fit],
                                  [float(row["target"]) for row in fit])
        calibrated = calibrator.predict([float(row["prediction"]) for row in held])
        for row, value in zip(held, calibrated):
            outputs.append({**row, "calibrated_prediction": float(value)})
        fold_reports[str(fold)] = {
            "fit_rows": len(fit), "evaluation_rows": len(held),
            "fit_group_count": len(fit_groups), "evaluation_group_count": len(held_groups),
            "raw": _metrics([row["prediction"] for row in held], [row["target"] for row in held]),
            "calibrated": _metrics(calibrated, [row["target"] for row in held]),
        }
    outputs.sort(key=lambda row: (str(row["dataset_id"]), str(row["image_id"])))
    target = [float(row["target"]) for row in outputs]
    report = {
        "schema_version": "shared-monotonic-oof-evaluation-v1",
        "locked_test_accessed": False, "dataset_identity_used_as_input": False,
        "cross_fit_folds": folds, "folds": fold_reports,
        "overall": {
            "raw": _metrics([row["prediction"] for row in outputs], target),
            "calibrated": _metrics([row["calibrated_prediction"] for row in outputs], target),
        }, "per_dataset": {},
    }
    for dataset in sorted({str(row["dataset_id"]) for row in outputs}):
        selected = [row for row in outputs if str(row["dataset_id"]) == dataset]
        y = [row["target"] for row in selected]
        report["per_dataset"][dataset] = {
            "raw": _metrics([row["prediction"] for row in selected], y),
            "calibrated": _metrics([row["calibrated_prediction"] for row in selected], y),
        }
    final = fit_isotonic([float(row["prediction"]) for row in rows],
                         [float(row["target"]) for row in rows])
    return report, final.payload(), outputs


def _atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.oof_predictions.read_text().splitlines() if line]
    report, calibrator, outputs = evaluate(rows)
    digest = hashlib.sha256(args.oof_predictions.read_bytes()).hexdigest()
    report["input_sha256"] = digest
    calibrator.update({"fit_source": "development OOF predictions only",
                       "input_sha256": digest, "locked_test_accessed": False})
    _atomic(args.output_dir / "calibration-report.json", report)
    _atomic(args.output_dir / "shared-calibrator.json", calibrator)
    prediction_path = args.output_dir / "cross-fitted-predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in outputs))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
