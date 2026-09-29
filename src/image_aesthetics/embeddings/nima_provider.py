"""Pinned pretrained NIMA provider for the tenth fusion signal."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from .feature_contract import DIMENSIONS, FEATURES

NIMA_PATH = Path("/Users/rushivyas/.cache/torch/hub/pyiqa/NIMA_InceptionV2_ava-b0c77c00.pth")
NIMA_SHA256 = "b0c77c00c789565be8e6af96c1641bbad716c1592bff1cdc749c6006b95bd34e"
NIMA_MODEL = "pyiqa-nima-inception_resnet_v2-ava-b0c77c00"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_nima_image(path: Path, image_size: int = 299) -> tuple[np.ndarray, np.ndarray, str]:
    """Strict decode followed by NIMA's aspect resize and centered 299px crop."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    width, height = image.size
    scale = image_size / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    left = (resized.width - image_size) // 2
    top = (resized.height - image_size) // 2
    crop = resized.crop((left, top, left + image_size, top + image_size))
    pixels = np.ascontiguousarray(crop, dtype=np.uint8)
    return pixels, pixels, "pillow-exif-rgb-nima-center-crop-v1"


class NimaExtractor:
    names = ("nima",)
    requires_native_images = False

    def __init__(self, *, device: str | None = None, model: Any | None = None):
        import torch

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        if not NIMA_PATH.is_file() or _sha256(NIMA_PATH) != NIMA_SHA256:
            raise RuntimeError(f"missing or invalid pinned NIMA checkpoint: {NIMA_PATH}")
        if model is None:
            import pyiqa
            self.model = pyiqa.create_metric(
                "nima", device=self.device, pretrained=False,
                pretrained_model_path=str(NIMA_PATH),
            )
        else:
            self.model = model
        self.model.eval()
        self.last_timings: dict[str, float] = {}

    def extract(self, images: np.ndarray) -> dict[str, np.ndarray]:
        count = len(images)
        if images.dtype != np.uint8 or images.shape != (count, 299, 299, 3):
            raise ValueError(f"NIMA input must be uint8 (N,299,299,3), got {images.shape} {images.dtype}")
        started = time.perf_counter()
        tensor = self.torch.from_numpy(images.transpose(0, 3, 1, 2).copy())
        tensor = tensor.to(self.device, dtype=self.torch.float32).div_(255.0)
        with self.torch.inference_mode():
            raw = self.model(tensor).reshape(-1).float()
            normalized = ((raw - 1.0) / 9.0).clamp(0.0, 1.0)
        self.last_timings["nima"] = time.perf_counter() - started
        return {"nima": normalized.cpu().numpy().astype(np.float32, copy=False)[:, None]}


def nima_contract() -> dict[str, Any]:
    """Fixed ten-signal schema with NIMA active and all other signals inactive."""
    features: dict[str, dict[str, Any]] = {}
    for name in FEATURES:
        if name == "nima":
            features[name] = {
                "dimension": 1,
                "dtype": "float32",
                "range": [0.0, 1.0],
                "direction": "higher_is_better",
                "model_revision": NIMA_MODEL,
                "checkpoint_sha256": NIMA_SHA256,
                "preprocessing": {
                    "decoder": "Pillow",
                    "orientation": "EXIF transpose",
                    "color": "RGB",
                    "resize": "shorter side to 299 preserving aspect ratio",
                    "crop": "center 299x299",
                    "interpolation": "bilinear",
                    "input": "float32 uint8/255",
                    "model_normalization": "PyIQA/timm ImageNet mean and std",
                },
                "normalization": "clip((expected AVA rating 1..10 - 1) / 9, 0, 1)",
            }
        else:
            identity = f"inactive-nima-only:{name}"
            features[name] = {
                "dimension": DIMENSIONS.get(name, 1), "dtype": "float32",
                "range": [-1.0, 1.0] if name in DIMENSIONS else [0.0, 1.0],
                "direction": "higher_is_better", "model_revision": identity,
                "checkpoint_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "preprocessing": {"status": "not_generated_by_nima_only"},
                "normalization": "not_generated_by_nima_only",
            }
    return {"features": features}
