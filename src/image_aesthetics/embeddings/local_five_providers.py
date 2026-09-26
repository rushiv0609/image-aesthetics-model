"""Strict locally available providers for the partial ten-signal smoke.

This module intentionally implements only signals whose exact local artifacts
are known.  It never substitutes defaults for an unavailable neural scorer.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .backend import SigLIP2Embedder

SIGLIP_MODEL = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP_PATH = Path("/Users/rushivyas/.cache/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/model.safetensors")
SIGLIP_SHA256 = "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
A1_PATH = Path("/Users/rushivyas/datasets/image-aesthetics/ava/releases/siglip2-aesthetic/a1-v1/model.safetensors")
A1_SHA256 = "b921e8568db6229a5a89810dd5c5823e755261acb9d0e91eb83fc35f594aabf1"
LOCAL_FIVE = ("siglip_embedding", "a1_siglip", "sharpness", "contrast", "colorfulness")


def _check_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required checkpoint missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"checkpoint hash mismatch: {path}")


def scalar_metrics(images: Any) -> dict[str, np.ndarray]:
    """Docs-compatible formulas; input is uint8 RGB BHWC.

    OpenCV's 8-bit ``RGB2GRAY`` path uses 15-bit fixed-point coefficients
    ``(9798, 19235, 3735)`` with a rounding offset of ``1 << 14``. Preserve
    that integer operation exactly before evaluating the default 3x3
    ``Laplacian(CV_64F)`` 4-neighbour kernel in float64 with
    BORDER_REFLECT_101 (``np.pad(..., mode='reflect')``).
    """
    x = np.asarray(images)
    if x.ndim != 4 or x.shape[-1] != 3 or x.dtype != np.uint8:
        raise ValueError(f"expected uint8 (N,H,W,3) RGB images, got {x.shape} {x.dtype}")
    rgb32 = x.astype(np.int32)
    r, g, b = rgb32[..., 0], rgb32[..., 1], rgb32[..., 2]
    gray = ((9798 * r + 19235 * g + 3735 * b + (1 << 14)) >> 15).astype(np.uint8)
    gray64 = gray.astype(np.float64)
    padded = np.pad(gray64, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    lap = (padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] + padded[:, 1:-1, :-2] +
           padded[:, 1:-1, 2:] - 4.0 * gray64)
    sharpness = np.clip(lap.var(axis=(1, 2), dtype=np.float64) / 1200.0, 0.0, 1.0)
    p5, p95 = np.percentile(gray, (5, 95), axis=(1, 2))
    contrast = np.clip((p95 - p5) / 220.0, 0.0, 1.0)
    pixels32=x.astype(np.float32)
    rc, gc, bc=pixels32[...,0],pixels32[...,1],pixels32[...,2]
    rg, yb = np.abs(rc - gc), np.abs(0.5 * (rc + gc) - bc)
    metric = np.sqrt(rg.std(axis=(1, 2)) ** 2 + yb.std(axis=(1, 2)) ** 2)
    metric += 0.3 * np.sqrt(rg.mean(axis=(1, 2)) ** 2 + yb.mean(axis=(1, 2)) ** 2)
    colorfulness = np.clip(metric / 100.0, 0.0, 1.0)
    return {"sharpness": sharpness[:, None].astype(np.float32), "contrast": contrast[:, None].astype(np.float32),
            "colorfulness": colorfulness[:, None].astype(np.float32)}


class LocalFiveExtractor:
    """One SigLIP pass, then derive A1 and scalar metrics from that batch."""
    names = LOCAL_FIVE

    def __init__(self, *, siglip: Any | None = None, a1_path: Path = A1_PATH, a1: Any | None = None):
        import torch
        self._torch = torch
        if a1 is None:
            _check_file(a1_path, A1_SHA256)
            from safetensors.torch import load_file
            from image_aesthetics.models.siglip_head.models import build_head
            self.a1 = build_head("A1")
            self.a1.load_state_dict(load_file(str(a1_path)), strict=True)
            self.a1.eval()
        else:
            self.a1 = a1
        if siglip is None:
            _check_file(SIGLIP_PATH, SIGLIP_SHA256)
        self.siglip = siglip or SigLIP2Embedder.production(SIGLIP_MODEL, SIGLIP_REVISION)

    def extract(self, images: Any) -> dict[str, np.ndarray]:
        started = time.perf_counter()
        embeddings = self.siglip.embed(images)
        siglip_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with self._torch.inference_mode():
            logits = self.a1(self._torch.from_numpy(embeddings)).cpu().numpy()
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        score = ((probs @ np.arange(1, 11, dtype=np.float32) - 1.0) / 9.0).clip(0, 1)
        a1_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = {"siglip_embedding": embeddings.astype(np.float32, copy=False), "a1_siglip": score[:, None].astype(np.float32)}
        result.update(scalar_metrics(images))
        scalar_seconds = time.perf_counter() - started
        self.last_timings = {"siglip_embedding": siglip_seconds, "a1_siglip": a1_seconds,
                             "sharpness": scalar_seconds / 3, "contrast": scalar_seconds / 3,
                             "colorfulness": scalar_seconds / 3}
        return result


def local_five_contract_metadata() -> dict[str, dict[str, Any]]:
    # Shared decode describes pixels only. Model normalization belongs to the
    # SigLIP branch; scalar formulas operate directly on these decoded bytes.
    decoded_rgb = {
        "image_size": 224, "decoder": "Pillow", "orientation": "EXIF transpose",
        "color": "RGB", "pixel_dtype": "uint8", "pixel_layout": "HWC",
        "resize": "aspect-preserving thumbnail", "interpolation": "Lanczos",
        "padding": "black centered square canvas",
    }
    siglip_pixels = {
        **decoded_rgb,
        "model_normalization": {
            "transform": "float32(pixel) / 127.5 - 1.0",
            "equivalent_mean": [0.5, 0.5, 0.5],
            "equivalent_std": [0.5, 0.5, 0.5],
            "input_range": [0, 255], "output_range": [-1.0, 1.0],
        },
    }
    signal = {
        "siglip_embedding": {
            "model_revision": f"{SIGLIP_MODEL}@{SIGLIP_REVISION}",
            "checkpoint_sha256": SIGLIP_SHA256,
            "dtype": "float32", "range": [-1.0, 1.0], "direction": "higher_is_better",
            "preprocessing": siglip_pixels,
            "normalization": "L2 unit norm across 768 dimensions; atol=0.005",
        },
        "a1_siglip": {
            "model_revision": "AVA-A1-head-v1-on-SigLIP2-Base-224",
            "checkpoint_sha256": A1_SHA256,
            "dtype": "float32", "range": [0.0, 1.0], "direction": "higher_is_better",
            "preprocessing": {"input_signal": "siglip_embedding", "input_dtype": "float32",
                              "input_dimension": 768, "input_normalization": "L2 unit norm across 768 dimensions",
                              "checkpoint": str(A1_PATH)},
            "normalization": "softmax expected rating (1..10), then (expected_rating - 1) / 9",
        },
    }
    formulas = {
        "sharpness": {
            "formula": "variance(default 3x3 Laplacian of RGB2GRAY uint8) / 1200, clipped [0,1]",
            "model_revision": "numpy-opencv-compatible-rgb2gray15-laplacian-cv64f-v1",
            "normalization": "population variance / 1200, clipped [0,1]",
            "preprocessing": {**decoded_rgb, "input": "raw decoded uint8 RGB pixels",
                              "grayscale": "OpenCV 15-bit fixed-point RGB2GRAY",
                              "laplacian": "4-neighbor CV_64F, BORDER_REFLECT_101"},
        },
        "contrast": {
            "formula": "(gray_p95 - gray_p5) / 220, clipped [0,1]",
            "model_revision": "numpy-opencv-compatible-rgb2gray15-percentile-contrast-v1",
            "normalization": "p95 minus p5 / 220, NumPy linear percentile, clipped [0,1]",
            "preprocessing": {**decoded_rgb, "input": "raw decoded uint8 RGB pixels",
                              "grayscale": "OpenCV 15-bit fixed-point RGB2GRAY",
                              "percentile": "NumPy percentile method=linear on quantized uint8 gray"},
        },
        "colorfulness": {
            "formula": "Hasler-Susstrunk opponent-color metric / 100, clipped [0,1]",
            "model_revision": "numpy-hasler-susstrunk-rgb-float32-v1",
            "normalization": "Hasler-Susstrunk metric / 100, clipped [0,1]",
            "preprocessing": {**decoded_rgb, "input": "raw decoded uint8 RGB pixels",
                              "working_dtype": "float32 RGB; no model input normalization"},
        },
    }
    for name, descriptor in formulas.items():
        descriptor["dtype"] = "float32"
        descriptor["range"] = [0.0, 1.0]
        descriptor["direction"] = "higher_is_better"
        descriptor["checkpoint_sha256"] = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        signal[name] = descriptor
    return signal
