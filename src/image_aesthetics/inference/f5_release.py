"""Strict loader for the versioned three-seed F5-G release bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from image_aesthetics.data.normalization import FeatureNormalizer
from image_aesthetics.evaluation.monotonic_calibration import IsotonicCalibrator
from image_aesthetics.models.fusion import build_baseline
from image_aesthetics.models.fusion.contract import SCALAR_NAMES, flatten_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class F5ReleaseScorer:
    """Load one immutable F5-G bundle and return raw or calibrated scores.

    The vision backbones are intentionally outside this class. Moments Generator
    supplies the already-computed SigLIP 2 and CLIP embeddings plus eight scalar
    signals, preserving the ingestion-time cache contract.
    """

    def __init__(self, release_dir: Path | str, *, device: str = "cpu",
                 verify_hashes: bool = True):
        self.release_dir = Path(release_dir)
        manifest = json.loads((self.release_dir / "manifest.json").read_text())
        if manifest.get("schema_version") != "f5-g-release-manifest-v1":
            raise ValueError("unsupported F5 release manifest")
        if verify_hashes:
            for name, expected in manifest.get("files", {}).items():
                path = self.release_dir / name
                if not path.is_file() or _sha256(path) != expected:
                    raise ValueError(f"release file failed SHA-256 verification: {name}")

        config = json.loads((self.release_dir / "model-config.json").read_text())
        if (config.get("architecture", {}).get("model_id") != "F5-G-v1"
                or config.get("feature_contract") != "joined-ten-v1"
                or config.get("ensemble") != "mean-of-three-seeds"):
            raise ValueError("release model contract is incompatible")
        self.device = torch.device(device)
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        self.normalizer = FeatureNormalizer.from_file(self.release_dir / "normalization.json")

        calibration = json.loads((self.release_dir / "calibration.json").read_text())
        if calibration.get("schema_version") != "shared-isotonic-v1":
            raise ValueError("release calibrator is incompatible")
        self.calibrator = IsotonicCalibrator(tuple(map(float, calibration["thresholds"])),
                                              tuple(map(float, calibration["values"])))

        from safetensors.torch import load_file
        self.models = []
        for member in config["members"]:
            model = build_baseline("F5-G", **config["architecture"].get("kwargs", {}))
            model.load_state_dict(load_file(str(self.release_dir / member["file"])), strict=True)
            model.to(self.device).eval()
            self.models.append(model)
        if len(self.models) != 3:
            raise ValueError("F5-G production release requires exactly three members")

    @staticmethod
    def feature_tensor(siglip_embedding: Sequence[float], clip_embedding: Sequence[float],
                       scalars: Mapping[str, float]) -> torch.Tensor:
        """Create the immutable 1,544-float feature vector from named signals."""
        missing = [name for name in SCALAR_NAMES if name not in scalars]
        extra = sorted(set(scalars) - set(SCALAR_NAMES))
        if missing or extra:
            raise ValueError(f"scalar contract mismatch; missing={missing}, extra={extra}")
        siglip = torch.as_tensor(siglip_embedding, dtype=torch.float32)
        clip = torch.as_tensor(clip_embedding, dtype=torch.float32)
        scalar = torch.tensor([float(scalars[name]) for name in SCALAR_NAMES], dtype=torch.float32)
        if siglip.shape != (768,) or clip.shape != (768,):
            raise ValueError("SigLIP 2 and CLIP embeddings must each have shape (768,)")
        if not torch.isfinite(siglip).all() or not torch.isfinite(clip).all() or not torch.isfinite(scalar).all():
            raise ValueError("input signals contain non-finite values")
        return flatten_features(siglip, clip, scalar)

    def score_tensor(self, features: torch.Tensor, *, calibrated: bool = True) -> float | np.ndarray:
        """Score one feature row or a batch; calibrated output is always [0,1]."""
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != 1544 or not torch.isfinite(tensor).all():
            raise ValueError("features must be finite with shape (1544,) or (N,1544)")
        normalized = self.normalizer.transform(tensor)
        with torch.inference_mode():
            raw = torch.stack([model(normalized) for model in self.models]).mean(0).cpu().numpy()
        values = self.calibrator.predict(raw) if calibrated else raw.astype(np.float64)
        values = np.clip(values, 0.0, 1.0)
        return float(values[0]) if single else values

    def score(self, siglip_embedding: Sequence[float], clip_embedding: Sequence[float],
              scalars: Mapping[str, float], *, calibrated: bool = True) -> float:
        return float(self.score_tensor(self.feature_tensor(siglip_embedding, clip_embedding, scalars),
                                       calibrated=calibrated))
