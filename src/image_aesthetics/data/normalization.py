"""Leakage-safe streaming feature normalization for fusion folds.

Stats are fitted from an iterator (never by materializing a full feature
matrix), and persisted atomically with enough identity metadata to reject a
stats file fitted for another split, fold, or feature contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "fusion-normalization-v1"


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - dependency check
        raise RuntimeError("feature normalization requires NumPy") from exc
    return np


def _as_array(value: Any):
    np = _numpy()
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        array = array.reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("normalization input contains NaN or Inf")
    return array


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NormalizationStats:
    """Population mean/std and provenance for one training partition."""

    feature_dim: int
    count: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    identity: Mapping[str, Any]
    epsilon: float = 1e-6
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.feature_dim < 1 or self.count < 1:
            raise ValueError("normalization stats require positive dimension and count")
        if len(self.mean) != self.feature_dim or len(self.std) != self.feature_dim:
            raise ValueError("normalization vector dimension mismatch")
        if self.epsilon <= 0 or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite and positive")
        np = _numpy()
        if not np.isfinite(np.asarray(self.mean)).all() or not np.isfinite(np.asarray(self.std)).all():
            raise ValueError("normalization stats contain NaN or Inf")
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization std must be positive")

    @property
    def identity_hash(self) -> str:
        return _hash(dict(self.identity))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_dim": self.feature_dim,
            "count": self.count,
            "mean": list(self.mean),
            "std": list(self.std),
            "epsilon": self.epsilon,
            "identity": dict(self.identity),
            "identity_hash": self.identity_hash,
        }

    @property
    def stats_hash(self) -> str:
        return _hash(self.payload())

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NormalizationStats":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported normalization stats schema")
        identity = payload.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError("normalization stats missing identity metadata")
        if payload.get("identity_hash") != _hash(dict(identity)):
            raise ValueError("normalization identity hash mismatch")
        stats = cls(
            feature_dim=int(payload["feature_dim"]), count=int(payload["count"]),
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            identity=dict(identity), epsilon=float(payload.get("epsilon", 1e-6)),
        )
        expected_hash = payload.get("stats_hash")
        if expected_hash is not None and expected_hash != stats.stats_hash:
            raise ValueError("normalization stats hash mismatch")
        return stats


def fit_streaming(features: Iterable[Any], *, identity: Mapping[str, Any],
                  epsilon: float = 1e-6, stats_path: Path | None = None,
                  feature_key: str | None = None) -> NormalizationStats:
    """Fit population moments in one pass using vector Welford updates."""
    np = _numpy()
    count = 0
    mean = None
    m2 = None
    for value in features:
        if feature_key is not None:
            if not isinstance(value, Mapping) or feature_key not in value:
                raise ValueError(f"normalization sample is missing feature key: {feature_key}")
            value = value[feature_key]
        array = _as_array(value)
        if mean is None:
            mean = np.zeros(array.shape[0], dtype=np.float64)
            m2 = np.zeros(array.shape[0], dtype=np.float64)
        if array.shape[0] != mean.shape[0]:
            raise ValueError("normalization input dimensions are inconsistent")
        count += 1
        delta = array - mean
        mean += delta / count
        m2 += delta * (array - mean)
    if count == 0 or mean is None or m2 is None:
        raise ValueError("cannot fit normalization stats on an empty stream")
    variance = np.maximum(m2 / count, 0.0)
    std = np.sqrt(variance)
    std = np.where(std < epsilon, 1.0, std)
    stats = NormalizationStats(int(mean.shape[0]), count, tuple(mean.tolist()), tuple(std.tolist()), dict(identity), epsilon)
    if stats_path is not None:
        save_stats(stats_path, stats)
    return stats


def save_stats(path: Path, stats: NormalizationStats) -> None:
    """Persist stats atomically, including a checksum over the full payload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = stats.payload()
    payload["stats_hash"] = stats.stats_hash
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def load_stats(path: Path, *, expected_identity: Mapping[str, Any] | None = None) -> NormalizationStats:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stats = NormalizationStats.from_payload(payload)
    if expected_identity is not None and stats.identity_hash != _hash(dict(expected_identity)):
        raise ValueError("normalization stats identity does not match requested training partition")
    return stats


class FeatureNormalizer:
    """Transform NumPy arrays or torch tensors using persisted fold stats."""

    def __init__(self, stats: NormalizationStats):
        self.stats = stats

    @classmethod
    def fit(cls, features: Iterable[Any], *, identity: Mapping[str, Any],
            epsilon: float = 1e-6, stats_path: Path | None = None,
            feature_key: str | None = None) -> "FeatureNormalizer":
        return cls(fit_streaming(features, identity=identity, epsilon=epsilon,
                                 stats_path=stats_path, feature_key=feature_key))

    @classmethod
    def from_file(cls, path: Path, *, expected_identity: Mapping[str, Any] | None = None) -> "FeatureNormalizer":
        return cls(load_stats(path, expected_identity=expected_identity))

    def transform(self, features: Any) -> Any:
        np = _numpy()
        if hasattr(features, "detach"):
            # Preserve torch dtype/device and avoid a host round-trip.
            import torch
            mean = torch.as_tensor(self.stats.mean, dtype=features.dtype, device=features.device)
            std = torch.as_tensor(self.stats.std, dtype=features.dtype, device=features.device)
            return (features - mean) / std
        array = np.asarray(features)
        mean = np.asarray(self.stats.mean, dtype=array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32)
        std = np.asarray(self.stats.std, dtype=mean.dtype)
        return (array - mean) / std

    __call__ = transform


# Descriptive aliases used by callers that prefer the streaming terminology.
StreamingFeatureNormalizer = FeatureNormalizer
persist_stats = save_stats
read_stats = load_stats
fit_normalization_stats = fit_streaming
fit_fold_normalization = fit_streaming
