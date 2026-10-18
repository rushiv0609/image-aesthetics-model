"""Shared, dataset-agnostic isotonic calibration with OOF provenance checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Piecewise-constant non-decreasing mapping fitted by PAV."""

    thresholds: tuple[float, ...]
    values: tuple[float, ...]

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        x = np.asarray(scores, dtype=np.float64)
        indexes = np.searchsorted(np.asarray(self.thresholds), x, side="left")
        indexes = np.clip(indexes, 0, len(self.values) - 1)
        return np.asarray(self.values)[indexes]

    def payload(self) -> dict[str, object]:
        return {"schema_version": "shared-isotonic-v1",
                "thresholds": list(self.thresholds), "values": list(self.values)}


def fit_isotonic(scores: Sequence[float], targets: Sequence[float]) -> IsotonicCalibrator:
    """Fit a single shared monotonic map using pool-adjacent-violators."""
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("scores and targets must be non-empty matching vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("calibration data contains non-finite values")
    if ((x < 0) | (x > 1) | (y < 0) | (y > 1)).any():
        raise ValueError("shared calibration expects normalized [0,1] values")
    order = np.argsort(x, kind="mergesort")
    sx, sy = x[order], y[order]
    # Equal scores must share one fitted value. If they remain separate,
    # searchsorted(side="left") silently selects just the first tied sample.
    unique_scores, inverse, counts = np.unique(sx, return_inverse=True,
                                               return_counts=True)
    target_sums = np.bincount(inverse, weights=sy)
    blocks: list[list[float]] = []  # max_x, weight, mean
    for score, count, target_sum in zip(unique_scores, counts, target_sums):
        blocks.append([float(score), float(count), float(target_sum / count)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop(); left = blocks.pop()
            weight = left[1] + right[1]
            mean = (left[1] * left[2] + right[1] * right[2]) / weight
            blocks.append([right[0], weight, mean])
    return IsotonicCalibrator(tuple(block[0] for block in blocks),
                              tuple(float(np.clip(block[2], 0, 1)) for block in blocks))


def validate_oof_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Reject locked, in-fold, duplicate, or incomplete prediction records."""
    result = [dict(row) for row in rows]
    if not result:
        raise ValueError("OOF prediction set is empty")
    seen: set[tuple[str, str]] = set()
    for row in result:
        required = {"dataset_id", "image_id", "group_id", "development_fold",
                    "source_validation_fold", "prediction", "target", "locked_test"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"OOF row missing fields: {sorted(missing)}")
        if row["locked_test"] is not False:
            raise ValueError("locked-test rows are forbidden in calibration")
        if int(row["development_fold"]) != int(row["source_validation_fold"]):
            raise ValueError("prediction is not out-of-fold for its row")
        key = (str(row["dataset_id"]), str(row["image_id"]))
        if key in seen:
            raise ValueError(f"duplicate OOF prediction: {key}")
        seen.add(key)
    return result
