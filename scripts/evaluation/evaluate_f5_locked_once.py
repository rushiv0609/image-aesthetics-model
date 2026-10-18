#!/usr/bin/env python3
"""Plan or perform the single locked-test evaluation after frozen final fit.

The explicit ``--execute`` path atomically claims an external receipt before
opening the locked partition. Any existing receipt, including a failed or
interrupted attempt, permanently prevents a retry.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.evaluation.monotonic_calibration import IsotonicCalibrator
from image_aesthetics.models.fusion import build_baseline
from image_aesthetics.training.f5_finalization import (
    assert_external_path, atomic_json, claim_locked_evaluation,
    load_frozen_policy, sha256_file,
)
from scripts.training.run_f5_screen import _cache_hashes
from image_aesthetics.training.fusion_trainer import _metrics


def plan(policy_path: Path, fit_root: Path, receipt_path: Path) -> dict:
    policy, digest = load_frozen_policy(policy_path)
    return {"state": "planned", "policy_sha256": digest,
            "candidate": policy["candidate"], "seeds": policy["seeds"],
            "device": policy["training"]["device"],
            "fit_root": str(fit_root), "receipt": str(receipt_path),
            "will_claim_exactly_one_attempt": True,
            "locked_test_accessed": False}


def _canonical_receipt(data_root: Path) -> Path:
    """One stable external receipt path prevents alternate-path retries."""
    return Path(data_root).resolve() / "fusion/experiments/f5-finalization/locked-test-evaluation.receipt.json"


def _metrics_by_dataset(prediction: list[float], target: list[float], datasets: list[str]):
    result = {"overall": _metrics(prediction, target), "per_dataset": {}}
    for dataset in sorted(set(datasets)):
        indexes = [index for index, value in enumerate(datasets) if value == dataset]
        result["per_dataset"][dataset] = _metrics(
            [prediction[index] for index in indexes], [target[index] for index in indexes])
    return result


def _verify_fit(policy: dict, policy_digest: str, fit_root: Path) -> tuple[dict, str]:
    manifest_path = fit_root / "fit-manifest.json"
    status_path = fit_root / "fit-status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("successful full-development fit artifacts are required")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    status_digest = sha256_file(status_path)
    if (manifest.get("state") != "completed" or status.get("state") != "completed"
            or manifest.get("policy_sha256") != policy_digest
            or status.get("policy_sha256") != policy_digest
            or manifest.get("candidate") != policy["candidate"]
            or manifest.get("split_sha256") != policy["split_sha256"]
            or manifest.get("feature_hashes") != policy["feature_hashes"]
            or manifest.get("fit_manifest_sha256") != status_digest
            or set(manifest.get("seeds", {})) != {str(seed) for seed in policy["seeds"]}
            or status.get("locked_test_accessed") is not False
            or manifest.get("locked_test_accessed") is not False):
        raise ValueError("final-fit manifest is incomplete or conflicts with frozen policy")
    for seed in policy["seeds"]:
        seed_entry = manifest["seeds"][str(seed)]
        for stage, expected_epochs in (("f2", policy["epochs"]["f2"]),
                                       ("candidate", policy["epochs"]["candidate"])):
            entry = seed_entry.get(stage, {})
            checkpoint = Path(entry.get("checkpoint", ""))
            if (not checkpoint.is_file()
                    or sha256_file(checkpoint) != entry.get("checkpoint_sha256")
                    or int(entry.get("epochs", -1)) != int(expected_epochs)):
                raise ValueError(f"final {stage} checkpoint is missing or invalid for seed {seed}")
    return manifest, status_digest


def _verify_non_locked_inputs(policy: dict, manifest: dict, *, data_root: Path,
                              fit_root: Path) -> tuple[Path, dict[str, Path],
                                                       IsotonicCalibrator, FeatureNormalizer]:
    """Validate every artifact that can be checked without opening test rows.

    This deliberately runs before the irreversible receipt claim. A missing or
    corrupted cache, calibration file, or normalizer is an operational setup
    error, not a legitimate consumption of the one permitted locked evaluation.
    """
    split = data_root / "fusion/prepared/v6/samples.jsonl"
    join_dirs = {name: data_root / name / "features/joined-ten-v1"
                 for name in policy["training"]["datasets"]}
    if set(join_dirs) != set(policy["feature_hashes"]):
        raise ValueError("policy feature hashes do not exactly cover evaluation datasets")
    if not split.is_file() or sha256_file(split) != policy["split_sha256"]:
        raise ValueError("current split differs from frozen policy")
    if _cache_hashes(join_dirs) != policy["feature_hashes"]:
        raise ValueError("current joined features differ from frozen policy")

    calibration_path = Path(policy["calibration"]["path"]).resolve()
    assert_external_path(calibration_path, repo_root=ROOT, data_root=data_root)
    if (not calibration_path.is_file()
            or sha256_file(calibration_path) != policy["calibration"]["sha256"]):
        raise ValueError("shared calibrator is missing or differs from frozen policy")
    calibration = json.loads(calibration_path.read_text())
    if (calibration.get("schema_version") != "shared-isotonic-v1"
            or calibration.get("locked_test_accessed") is not False
            or calibration.get("fit_source") != "development OOF predictions only"
            or calibration.get("input_sha256") != policy["calibration"]["oof_predictions_sha256"]):
        raise ValueError("calibrator schema or provenance is invalid")
    thresholds = tuple(map(float, calibration["thresholds"]))
    values = tuple(map(float, calibration["values"]))
    if (not thresholds or len(thresholds) != len(values)
            or any(a > b for a, b in zip(thresholds, thresholds[1:]))
            or any(a > b for a, b in zip(values, values[1:]))
            or any(value < 0 or value > 1 for value in values)):
        raise ValueError("shared isotonic calibrator is malformed or non-monotonic")
    calibrator = IsotonicCalibrator(thresholds, values)

    normalizer_path = fit_root / "normalization.json"
    if (not normalizer_path.is_file()
            or sha256_file(normalizer_path) != manifest.get("normalization_sha256")):
        raise ValueError("full-development normalizer does not match fit manifest")
    expected_identity = {
        "schema_version": "fusion-training-view-v1",
        "split_manifest_sha256": policy["split_sha256"],
        "partition": "development",
        "validation_fold": None,
        "dataset_ids": sorted(join_dirs),
        "row_dataset_ids": sorted(join_dirs),
        "join_hashes": dict(sorted(policy["feature_hashes"].items())),
        "feature_dimension": 1544,
    }
    normalizer = FeatureNormalizer.from_file(
        normalizer_path, expected_identity=expected_identity)
    if normalizer.stats.stats_hash != manifest.get("normalization_stats_hash"):
        raise ValueError("normalizer statistics identity differs from fit manifest")
    return split, join_dirs, calibrator, normalizer


def execute(args) -> dict:
    policy, policy_digest = load_frozen_policy(args.policy)
    data_root = Path(args.data_root).resolve()
    fit_root = assert_external_path(args.fit_root, repo_root=ROOT, data_root=data_root)
    receipt_path = assert_external_path(_canonical_receipt(data_root),
                                       repo_root=ROOT, data_root=data_root)
    if args.receipt and Path(args.receipt).resolve() != receipt_path:
        raise ValueError("receipt path is fixed by the data root to enforce one global attempt")
    report_path = assert_external_path(args.report, repo_root=ROOT, data_root=data_root)
    if receipt_path.exists():
        raise FileExistsError(f"locked-test receipt already exists; refusing another attempt: {receipt_path}")
    if report_path.exists():
        raise FileExistsError(f"locked-test report path already exists: {report_path}")
    manifest, fit_status_sha = _verify_fit(policy, policy_digest, fit_root)
    split, join_dirs, calibrator, normalizer = _verify_non_locked_inputs(
        policy, manifest, data_root=data_root, fit_root=fit_root)

    # This O_EXCL durable receipt is the irreversible single-use gate. It is
    # written before any locked-test manifest, feature row, or label is opened.
    receipt = claim_locked_evaluation(
        receipt_path, policy_sha256=policy_digest,
        fit_manifest_sha256=sha256_file(fit_root / "fit-manifest.json"))
    receipt["fit_status_sha256"] = fit_status_sha
    try:
        test = FusionTrainingDataset(
            split, join_dirs, partition="locked_test", allow_locked_test=True,
            dataset_ids=policy["training"]["datasets"], normalizer=normalizer)
        if not len(test) or any(row.get("locked_test") is not True
                                for _ds, _index, row, _target in test.samples):
            raise ValueError("locked partition is empty or contains a non-test row")
        loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
        if args.device and args.device != policy["training"]["device"]:
            raise ValueError("requested device differs from the frozen policy")
        device = torch.device(policy["training"]["device"])
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        predictions = []; targets = []; dataset_ids = []
        for seed_index, seed in enumerate(policy["seeds"]):
            entry = manifest["seeds"][str(seed)]["candidate"]
            model = build_baseline(policy["candidate"], **policy["architecture"].get("kwargs", {}))
            state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"]); model.to(device); model.eval()
            seed_predictions = []
            with torch.no_grad():
                for batch in loader:
                    seed_predictions.append(model(batch["features"].to(device=device,
                                                                         dtype=torch.float32)).detach().cpu())
                    if seed_index == 0:
                        targets.extend(batch["target"].tolist())
                        dataset_ids.extend(map(str, batch["dataset_id"]))
            predictions.append(torch.cat(seed_predictions).numpy())
        raw = np.stack(predictions, axis=0).mean(axis=0).astype(float).tolist()
        calibrated = calibrator.predict(raw).astype(float).tolist()
        target = list(map(float, targets))
        result = {
            "schema_version": "f5-locked-evaluation-v1", "state": "completed",
            "candidate": policy["candidate"], "policy_sha256": policy_digest,
            "fit_manifest_sha256": sha256_file(fit_root / "fit-manifest.json"),
            "split_sha256": policy["split_sha256"],
            "feature_hashes": policy["feature_hashes"],
            "calibrator_sha256": policy["calibration"]["sha256"],
            "seeds": policy["seeds"], "rows": len(test),
            "raw": _metrics_by_dataset(raw, target, dataset_ids),
            "calibrated": _metrics_by_dataset(calibrated, target, dataset_ids),
            "locked_test_accessed": True, "evaluated_at_unix": time.time(),
        }
        atomic_json(report_path, result)
        receipt.update({"state": "completed", "report": str(report_path),
                        "report_sha256": sha256_file(report_path),
                        "completed_at_unix": time.time()})
        atomic_json(receipt_path, receipt)
        return result
    except BaseException as exc:
        receipt.update({"state": "failed_closed", "error_type": type(exc).__name__,
                        "error": str(exc), "failed_at_unix": time.time()})
        atomic_json(receipt_path, receipt)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="claim the receipt and evaluate once")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=None,
                        help="optional assertion of the fixed global receipt path")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=Path("/Users/rushivyas/datasets/image-aesthetics"))
    parser.add_argument("--device", choices=("mps", "cpu"), default=None,
                        help="optional assertion; evaluation uses the frozen policy device")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    assert_external_path(args.policy, repo_root=ROOT, data_root=args.data_root)
    receipt = _canonical_receipt(args.data_root)
    if args.receipt and Path(args.receipt).resolve() != receipt:
        raise ValueError("receipt path is fixed by the data root to enforce one global attempt")
    value = execute(args) if args.execute else plan(args.policy, args.fit_root, receipt)
    print(json.dumps(value, indent=2, sort_keys=True, default=str))
    return 0 if value["state"] in {"planned", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
