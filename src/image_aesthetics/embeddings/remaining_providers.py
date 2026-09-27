"""Locally pinned CLIP, LAION-CLIP and MUSIQ feature providers.

The AVA-trained NIMA checkpoint is intentionally not represented here: only an
ImageNet backbone is present locally, which is not a valid NIMA model.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from .feature_contract import DIMENSIONS, FEATURES

CLIP_ID = "hf-hub:timm/vit_large_patch14_clip_224.openai"
CLIP_SHA256 = "9ce2e8a8ebfff3793d7d375ad6d3c35cb9aebf3de7ace0fc7308accab7cd207e"
CLIP_PATH = Path("/Users/rushivyas/.cache/huggingface/hub/models--timm--vit_large_patch14_clip_224.openai/blobs") / CLIP_SHA256
LAION_PATH = Path("/Users/rushivyas/.cache/torch/hub/pyiqa/sac%2Blogos%2Bava1-l14-linearMSE.pth")
LAION_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
MUSIQ_AVA_PATH = Path("/Users/rushivyas/.cache/torch/hub/pyiqa/musiq_ava_ckpt-e8d3f067.pth")
MUSIQ_AVA_SHA256 = "e8d3f0671965dfe4301a1f7a4aedc8f49be8fed33b82dd5d116a9eb3ce2bd16b"
MUSIQ_KONIQ_PATH = Path("/Users/rushivyas/.cache/torch/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth")
MUSIQ_KONIQ_SHA256 = "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa"
NATIVE_MAX_SIDE = 512
REMAINING_FOUR = ("clip_embedding", "laion_clip", "musiq_aesthetic", "musiq_technical")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_checkpoint(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required pretrained checkpoint is missing: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"checkpoint SHA-256 mismatch for {path}: {actual}")


def prepare_remaining_images(path: Path, image_size: int = 224,
                             native_max_side: int = NATIVE_MAX_SIDE) -> tuple[np.ndarray, np.ndarray, str]:
    """Decode once for square CLIP and aspect-preserving bounded MUSIQ inputs."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
    square = image.copy()
    square.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(square, ((image_size - square.width) // 2,
                          (image_size - square.height) // 2))
    native = image.copy()
    native.thumbnail((native_max_side, native_max_side), Image.Resampling.LANCZOS)
    return (np.ascontiguousarray(canvas, dtype=np.uint8),
            np.ascontiguousarray(native, dtype=np.uint8), "pillow-exif-rgb-v1")


class _LAIONMLP:
    """Create the exact 768->1024->128->64->16->1 LAION scoring head."""

    @staticmethod
    def build():
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(768, 1024), nn.Dropout(0.2),
                    nn.Linear(1024, 128), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.Dropout(0.1),
                    nn.Linear(64, 16), nn.Linear(16, 1),
                )

            def forward(self, x):
                return self.layers(x)

        return Head()


class RemainingFourExtractor:
    """One CLIP pass feeds both CLIP signals; MUSIQ receives bounded native RGB."""
    names = REMAINING_FOUR
    requires_native_images = True

    def __init__(self, *, device: str | None = None, clip_model: Any | None = None,
                 laion_head: Any | None = None, musiq_ava: Any | None = None,
                 musiq_koniq: Any | None = None):
        import torch
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)

        _require_checkpoint(CLIP_PATH, CLIP_SHA256)
        if clip_model is None:
            import open_clip
            from safetensors.torch import load_file
            # The locally cached HF blob contains the OpenCLIP-format state
            # dict. Load it directly so provider startup is fully offline.
            self.clip_model = open_clip.create_model("ViT-L-14", pretrained=None, device="cpu")
            state = load_file(str(CLIP_PATH))
            self.clip_model.load_state_dict(state, strict=True)
            del state
            self.clip_model = self.clip_model.to(self.device)
            # Only the vision tower is used by this cache. Drop text-only
            # parameters after strict full-checkpoint validation to reduce RSS.
            for name in ("transformer", "token_embedding", "positional_embedding",
                         "ln_final", "text_projection", "logit_scale", "attn_mask"):
                if hasattr(self.clip_model, name):
                    setattr(self.clip_model, name, None)
            import gc
            gc.collect()
        else:
            self.clip_model = clip_model
        self.clip_model.eval()

        _require_checkpoint(LAION_PATH, LAION_SHA256)
        if laion_head is None:
            self.laion_head = _LAIONMLP.build()
            state = torch.load(str(LAION_PATH), map_location="cpu", weights_only=True)
            self.laion_head.load_state_dict(state, strict=True)
            self.laion_head = self.laion_head.to(self.device)
        else:
            self.laion_head = laion_head
        self.laion_head.eval()

        self.musiq_ava = musiq_ava or self._load_musiq(
            "musiq-ava", MUSIQ_AVA_PATH, MUSIQ_AVA_SHA256, num_class=10)
        self.musiq_koniq = musiq_koniq or self._load_musiq(
            "musiq", MUSIQ_KONIQ_PATH, MUSIQ_KONIQ_SHA256, num_class=1)
        self.last_timings: dict[str, float] = {}

    def _load_musiq(self, metric: str, checkpoint: Path, expected_sha: str, *, num_class: int):
        _require_checkpoint(checkpoint, expected_sha)
        import pyiqa
        model = pyiqa.create_metric(
            metric, as_loss=False, device=self.device, pretrained=False,
            pretrained_model_path=str(checkpoint), num_class=num_class,
            longer_side_lengths=[224, 384], max_seq_len_from_original_res=256,
        )
        # pyiqa's native MUSIQ path builds 224/384 aspect-preserving scales plus
        # an original-resolution stream; cap that stream at 256 patches.
        opts = model.net.data_preprocess_opts
        if opts.get("longer_side_lengths") != [224, 384] or opts.get("max_seq_len_from_original_res") != 256:
            raise RuntimeError(f"unexpected MUSIQ preprocessing for {metric}: {opts}")
        model.eval()
        return model

    def extract(self, images: np.ndarray, *, native_images: list[np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        count = len(images)
        if images.dtype != np.uint8 or images.shape != (count, 224, 224, 3):
            raise ValueError(f"CLIP input must be uint8 (N,224,224,3), got {images.shape} {images.dtype}")
        if len(native_images) != count or any(
            x.ndim != 3 or x.shape[2] != 3 or x.dtype != np.uint8 or
            min(x.shape[:2]) < 1 or max(x.shape[:2]) > NATIVE_MAX_SIDE
            for x in native_images
        ):
            raise ValueError(
                f"MUSIQ inputs must be one uint8 RGB HWC image per CLIP row "
                f"with longest side <= {NATIVE_MAX_SIDE}"
            )

        t0 = time.perf_counter()
        pixels = torch.from_numpy(images.transpose(0, 3, 1, 2).copy()).to(self.device, dtype=torch.float32) / 255.0
        pixels = (pixels - self.clip_mean.to(self.device)) / self.clip_std.to(self.device)
        with torch.inference_mode():
            embeddings = self.clip_model.encode_image(pixels)
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=-1)
        clip_np = embeddings.cpu().numpy().astype(np.float32, copy=False)
        if clip_np.shape != (count, 768):
            raise ValueError(f"CLIP returned unexpected shape {clip_np.shape}")
        self.last_timings["clip_embedding"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.inference_mode():
            raw = self.laion_head(embeddings).reshape(-1).float()
            laion = ((raw - 1.0) / 9.0).clamp(0.0, 1.0)
        laion_np = laion.cpu().numpy().astype(np.float32, copy=False)[:, None]
        self.last_timings["laion_clip"] = time.perf_counter() - t0

        musiq_aes = np.empty((count, 1), dtype=np.float32)
        musiq_tech = np.empty((count, 1), dtype=np.float32)
        aes_seconds = 0.0
        tech_seconds = 0.0
        with torch.inference_mode():
            for i, rgb in enumerate(native_images):
                tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
                tensor = tensor.to(self.device, dtype=torch.float32) / 255.0
                t0 = time.perf_counter()
                score = self.musiq_ava(tensor).reshape(-1)[0].float()
                musiq_aes[i, 0] = ((score - 1.0) / 9.0).clamp(0.0, 1.0).item()
                aes_seconds += time.perf_counter() - t0
                t0 = time.perf_counter()
                score = self.musiq_koniq(tensor).reshape(-1)[0].float()
                musiq_tech[i, 0] = (score / 100.0).clamp(0.0, 1.0).item()
                tech_seconds += time.perf_counter() - t0
        self.last_timings["musiq_aesthetic"] = aes_seconds
        self.last_timings["musiq_technical"] = tech_seconds
        return {"clip_embedding": clip_np, "laion_clip": laion_np,
                "musiq_aesthetic": musiq_aes, "musiq_technical": musiq_tech}


def remaining_four_contract_metadata() -> dict[str, dict[str, Any]]:
    common_224 = {
        "image_size": 224, "decoder": "Pillow", "orientation": "EXIF transpose",
        "color": "RGB", "pixel_dtype": "uint8", "pixel_layout": "HWC",
        "resize": "aspect-preserving thumbnail", "interpolation": "Lanczos",
        "padding": "black centered square canvas",
    }
    clip_input = {
        **common_224,
        "model_normalization": {
            "scale": "pixel / 255.0", "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
    }
    native_input = {
        "decoder": "Pillow", "orientation": "EXIF transpose", "color": "RGB",
        "pixel_dtype": "uint8", "pixel_layout": "HWC",
        "resize": f"aspect-preserving thumbnail; longest side <= {NATIVE_MAX_SIDE}",
        "interpolation": "Lanczos", "model_scales_long_side": [224, 384],
        "original_stream_max_patches": 256,
        "model_input_normalization": "float32(pixel) / 255.0; MUSIQ converts internally to [-1,1]",
    }
    values = {
        "clip_embedding": {
            "model_revision": CLIP_ID,
            "checkpoint_sha256": CLIP_SHA256,
            "preprocessing": clip_input,
            "normalization": "L2 unit norm across 768 dimensions",
            "range": [-1.0, 1.0],
        },
        "laion_clip": {
            "model_revision": "LAION-sac+logos+ava1-l14-linearMSE",
            "checkpoint_sha256": LAION_SHA256,
            "preprocessing": {"input_signal": "clip_embedding", "input_dimension": 768,
                              "input_normalization": "L2 unit norm"},
            "normalization": "clip((raw_score - 1) / 9, 0, 1)",
            "range": [0.0, 1.0],
        },
        "musiq_aesthetic": {
            "model_revision": "pyiqa-musiq-ava-e8d3f067",
            "checkpoint_sha256": MUSIQ_AVA_SHA256,
            "preprocessing": native_input,
            "normalization": "clip((MUSIQ AVA expected rating 1..10 - 1) / 9, 0, 1)",
            "range": [0.0, 1.0],
        },
        "musiq_technical": {
            "model_revision": "pyiqa-musiq-koniq10k-e95806b9",
            "checkpoint_sha256": MUSIQ_KONIQ_SHA256,
            "preprocessing": native_input,
            "normalization": "clip(raw KonIQ MUSIQ score / 100, 0, 1)",
            "range": [0.0, 1.0],
        },
    }
    for name, value in values.items():
        value["dtype"] = "float32"
        value["direction"] = "higher_is_better"
        value["dimension"] = DIMENSIONS.get(name, 1)
    return values


def remaining_contract() -> dict[str, Any]:
    """Return fixed ten-signal schema with six explicit inactive declarations."""
    from .local_five_providers import local_five_contract_metadata, LOCAL_FIVE

    local = local_five_contract_metadata()
    remaining = remaining_four_contract_metadata()
    features: dict[str, dict[str, Any]] = {}
    for name in FEATURES:
        if name in LOCAL_FIVE:
            source = local[name]
            features[name] = {"dimension": DIMENSIONS.get(name, 1), **{
                key: source[key] for key in (
                    "dtype", "range", "direction", "model_revision",
                    "checkpoint_sha256", "preprocessing", "normalization")}}
        elif name in REMAINING_FOUR:
            source = remaining[name]
            features[name] = {key: source[key] for key in (
                "dimension", "dtype", "range", "direction", "model_revision",
                "checkpoint_sha256", "preprocessing", "normalization")}
        else:
            features[name] = {
                "dimension": DIMENSIONS.get(name, 1), "dtype": "float32",
                "range": [-1.0, 1.0] if name in DIMENSIONS else [0.0, 1.0],
                "direction": "higher_is_better", "model_revision": f"inactive-remaining-four:{name}",
                "checkpoint_sha256": hashlib.sha256(f"inactive-remaining-four:{name}".encode()).hexdigest(),
                "preprocessing": {"status": "not_generated_by_remaining_four"},
                "normalization": "not_generated_by_remaining_four",
            }
    return {"features": features}
