"""Frozen-policy, full-development fit and fail-closed locked-test receipts."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


POLICY_SCHEMA = "f5-frozen-policy-v1"
HEX_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every decision required before final fitting or test access."""
    required = {
        "schema_version", "candidate", "epochs", "seeds", "calibration",
        "feature_hashes", "split_sha256", "thresholds", "evidence_hashes",
        "architecture", "preprocessing", "objective", "checkpoint_policy",
        "training",
    }
    missing = required - set(policy)
    if missing:
        raise ValueError(f"frozen policy missing fields: {sorted(missing)}")
    if policy["schema_version"] != POLICY_SCHEMA:
        raise ValueError("unsupported frozen-policy schema")
    if policy["candidate"] not in {"F5-R", "F5-G", "F5-G-NoGate", "F5-X"}:
        raise ValueError("frozen policy candidate is not an implemented F5 model")
    epochs = policy["epochs"]
    if not isinstance(epochs, Mapping) or set(epochs) != {"f2", "candidate"}:
        raise ValueError("epochs must explicitly freeze f2 and candidate epoch counts")
    if any(not isinstance(value, int) or value < 1 for value in epochs.values()):
        raise ValueError("frozen epoch counts must be positive integers")
    seeds = policy["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or any(not isinstance(seed, int) for seed in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError("seeds must be a non-empty list of unique integers")
    calibration = policy["calibration"]
    if (not isinstance(calibration, Mapping)
            or calibration.get("choice") != "shared-isotonic-v1"
            or not calibration.get("path")
            or not _is_sha(calibration.get("sha256"))
            or not _is_sha(calibration.get("oof_predictions_sha256"))):
        raise ValueError("calibration must identify the frozen shared isotonic artifact and hash")
    if not _is_sha(policy["split_sha256"]):
        raise ValueError("split_sha256 must be a SHA-256 digest")
    _hash_map(policy["feature_hashes"], "feature_hashes")
    _hash_map(policy["evidence_hashes"], "evidence_hashes")
    if not isinstance(policy["thresholds"], Mapping) or not policy["thresholds"]:
        raise ValueError("decision thresholds must be explicitly frozen")
    for name in ("architecture", "preprocessing", "objective", "checkpoint_policy"):
        if not isinstance(policy[name], Mapping) or not policy[name]:
            raise ValueError(f"{name} must be a non-empty frozen object")
    expected_architecture = {"F5-R": "F5-R-v1", "F5-G": "F5-G-v1",
                             "F5-G-NoGate": "F5-G-NoGate-v1", "F5-X": "F5-X-v1"}
    architecture = policy["architecture"]
    if architecture.get("model_id") != expected_architecture[policy["candidate"]]:
        raise ValueError("architecture model_id does not match the selected candidate")
    if "kwargs" in architecture and not isinstance(architecture["kwargs"], Mapping):
        raise ValueError("architecture kwargs must be an object")
    preprocessing = policy["preprocessing"]
    if preprocessing.get("feature_contract") != "joined-ten-v1":
        raise ValueError("final fit requires the frozen joined-ten-v1 preprocessing contract")
    objective = policy["objective"]
    if (objective.get("aesthetic_loss") != "calibrated-huber"
            or objective.get("ranking_loss") != "compatible-pairwise"):
        raise ValueError("final fit supports only the frozen calibrated-Huber/pairwise objective")
    checkpoint_policy = policy["checkpoint_policy"]
    if (checkpoint_policy.get("mode") != "fixed-epoch-final"
            or checkpoint_policy.get("early_stopping") is not False):
        raise ValueError("final checkpoint policy must use fixed epochs without early stopping")
    training = policy["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object")
    for field in ("batch_size", "datasets", "ranking_weight", "ranking_min_target_gap", "device"):
        if field not in training:
            raise ValueError(f"training is missing frozen field {field}")
    if (int(training["batch_size"]) < 1 or not isinstance(training["datasets"], list)
            or not training["datasets"]
            or len(training["datasets"]) != len(set(training["datasets"]))):
        raise ValueError("training batch size and dataset list must be non-empty")
    if training["device"] not in {"mps", "cpu"}:
        raise ValueError("frozen training device must be mps or cpu")
    if float(training["ranking_weight"]) < 0 or float(training["ranking_min_target_gap"]) < 0:
        raise ValueError("ranking loss parameters must be non-negative")
    return dict(policy)


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == HEX_LENGTH
            and all(char in "0123456789abcdef" for char in value.lower()))


def _hash_map(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty object")
    if any(not _is_sha(digest) for digest in value.values()):
        raise ValueError(f"{name} values must be SHA-256 digests")


def load_frozen_policy(path: Path) -> tuple[dict[str, Any], str]:
    """Load a policy whose adjacent digest sidecar has not changed."""
    path = Path(path)
    policy = validate_policy(json.loads(path.read_text()))
    digest = sha256_file(path)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    expected = digest_path.read_text().strip().split()[0]
    if expected != digest:
        raise ValueError("frozen policy digest does not match its immutable sidecar")
    return policy, digest


def freeze_policy(path: Path, policy: Mapping[str, Any]) -> str:
    """Create policy and SHA-256 sidecar exactly once; never overwrite either."""
    path = Path(path)
    payload = validate_policy(policy)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_durable_write(path, encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        _exclusive_durable_write(sidecar, f"{digest}  {path.name}\n".encode())
        path.chmod(0o444)
        sidecar.chmod(0o444)
    except Exception:
        # A policy without a matching sidecar cannot be consumed by either phase.
        raise
    return digest


def _exclusive_durable_write(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        # Preserve a partial/existing file as evidence; callers must choose a new path.
        raise


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def claim_locked_evaluation(receipt_path: Path, *, policy_sha256: str,
                            fit_manifest_sha256: str) -> dict[str, Any]:
    """Durably claim the sole evaluation attempt before opening locked data."""
    receipt = {
        "schema_version": "f5-locked-evaluation-receipt-v1",
        "state": "started", "policy_sha256": policy_sha256,
        "fit_manifest_sha256": fit_manifest_sha256,
        "started_at_unix": time.time(),
        "locked_test_access_started": True,
    }
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    _exclusive_durable_write(Path(receipt_path),
                             (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return receipt


def assert_external_path(path: Path, *, repo_root: Path, data_root: Path) -> Path:
    """Keep frozen/fitted/locked artifacts outside version control and source data."""
    resolved = Path(path).resolve()
    repo = Path(repo_root).resolve()
    data = Path(data_root).resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError("finalization artifact must be outside the Git repository")
    if resolved == data or resolved in data.parents:
        raise ValueError("finalization artifact must not replace or contain the dataset root")
    return resolved
