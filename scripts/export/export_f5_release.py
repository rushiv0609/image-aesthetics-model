#!/usr/bin/env python3
"""Export the frozen final F5-G ensemble into a Git-safe release bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalization-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable release path already exists: {args.output}")

    root = args.finalization_root
    policy_path = root / "frozen-policy-v1.json"
    fit_root = root / "final-fit-v1"
    fit = json.loads((fit_root / "fit-manifest.json").read_text())
    policy = json.loads(policy_path.read_text())
    report_path = root / "locked-test-report-v1.json"
    receipt_path = root / "locked-test-evaluation.receipt.json"
    report = json.loads(report_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (fit.get("state") != "completed" or report.get("state") != "completed"
            or receipt.get("state") != "completed" or policy.get("candidate") != "F5-G"
            or fit.get("locked_test_accessed") is not False
            or report.get("locked_test_accessed") is not True
            or report.get("policy_sha256") != fit.get("policy_sha256")):
        raise ValueError("finalization artifacts are incomplete or inconsistent")

    args.output.mkdir(parents=True, exist_ok=False)
    members = []
    for seed in policy["seeds"]:
        source = Path(fit["seeds"][str(seed)]["candidate"]["checkpoint"])
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        tensors = {name: value.detach().cpu().contiguous()
                   for name, value in checkpoint["model"].items()}
        name = f"model-seed-{seed}.safetensors"
        save_file(tensors, str(args.output / name), metadata={"seed": str(seed),
                  "architecture": "F5-G-v1", "format": "pt"})
        members.append({"seed": seed, "file": name,
                        "source_checkpoint_sha256": fit["seeds"][str(seed)]["candidate"]["checkpoint_sha256"]})

    shutil.copy2(fit_root / "normalization.json", args.output / "normalization.json")
    calibration_source = Path(policy["calibration"]["path"])
    shutil.copy2(calibration_source, args.output / "calibration.json")
    config = {"schema_version": "f5-g-model-config-v1", "model_version": "f5-g-v1",
              "architecture": policy["architecture"], "feature_contract": "joined-ten-v1",
              "feature_dimension": 1544, "scalar_order": policy["architecture"]["scalar_order"],
              "ensemble": "mean-of-three-seeds", "members": members,
              "outputs": {"raw": "ensemble mean", "production": "shared isotonic calibrated [0,1]"}}
    write_json(args.output / "model-config.json", config)
    write_json(args.output / "feature-contract.json", {
        "schema_version": "joined-ten-v1", "dimension": 1544,
        "blocks": {"siglip2_embedding": [0, 768], "local_scalars": [768, 772],
                   "clip_embedding": [772, 1540], "remaining_scalars": [1540, 1543],
                   "nima": [1543, 1544]},
        "scalar_order": policy["architecture"]["scalar_order"],
        "query_dependent_signals": False})
    write_json(args.output / "metrics.json", {
        "schema_version": "f5-g-release-metrics-v1", "locked_test_rows": report["rows"],
        "raw": report["raw"], "calibrated": report["calibrated"],
        "moments_human_benchmark": "pending real raters"})
    write_json(args.output / "provenance.json", {
        "schema_version": "f5-g-release-provenance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": fit["policy_sha256"], "policy_file_sha256": sha256(policy_path),
        "fit_manifest_sha256": sha256(fit_root / "fit-manifest.json"),
        "locked_report_sha256": sha256(report_path), "locked_receipt_sha256": sha256(receipt_path),
        "normalization_source_sha256": fit["normalization_sha256"],
        "calibration_source_sha256": policy["calibration"]["sha256"],
        "test_access": "single completed evaluation after immutable policy and final fit",
        "private_data_in_bundle": False})
    shutil.copy2(policy_path, args.output / "frozen-policy.json")

    tracked = sorted(path.name for path in args.output.iterdir())
    files = {name: sha256(args.output / name) for name in tracked}
    manifest = {"schema_version": "f5-g-release-manifest-v1", "model_version": "f5-g-v1",
                "files": files}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"release": str(args.output), "files": len(files) + 1,
                      "manifest_sha256": sha256(args.output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
