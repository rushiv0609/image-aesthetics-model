"""Immutable feature ordering for the production ten-signal contract."""
from __future__ import annotations
from typing import Mapping
import torch

EMBEDDING_DIM = 768
SCALAR_DIM = 8
FEATURE_DIM = 2 * EMBEDDING_DIM + SCALAR_DIM
# joined-ten-v1 is the concatenation of local-five (772), remaining-four
# (771), and NIMA (1).  The blocks are deliberately not reordered on disk.
SIGLIP_SLICE = slice(0, 768)
CLIP_SLICE = slice(772, 1540)
SCALAR_NAMES = ("a1_siglip", "laion_clip", "nima", "musiq_aesthetic",
                "musiq_technical", "sharpness", "contrast", "colorfulness")
SCALAR_INDICES = (768, 1540, 1543, 1541, 1542, 769, 770, 771)
SCALAR_INDEX = {name: i for i, name in enumerate(SCALAR_NAMES)}

def split_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"expected final dimension {FEATURE_DIM}, got {features.shape[-1]}")
    indices = torch.as_tensor(SCALAR_INDICES, device=features.device)
    return features[..., SIGLIP_SLICE], features[..., CLIP_SLICE], features.index_select(-1, indices)

def flatten_features(siglip: torch.Tensor, clip: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    if siglip.shape[-1] != EMBEDDING_DIM or clip.shape[-1] != EMBEDDING_DIM or scalars.shape[-1] != SCALAR_DIM:
        raise ValueError("expected SigLIP 768, CLIP 768, and scalar 8 feature blocks")
    output = torch.empty((*siglip.shape[:-1], FEATURE_DIM), dtype=siglip.dtype, device=siglip.device)
    output[..., SIGLIP_SLICE] = siglip
    output[..., 768:772] = scalars[..., (0, 5, 6, 7)]
    output[..., CLIP_SLICE] = clip
    output[..., 1540:1543] = scalars[..., (1, 3, 4)]
    output[..., 1543] = scalars[..., 2]
    return output

def scalar_vector(values: Mapping[str, float], *, dtype=torch.float32) -> torch.Tensor:
    missing = [name for name in SCALAR_NAMES if name not in values]
    if missing: raise ValueError(f"missing scalar signals: {missing}")
    return torch.tensor([values[name] for name in SCALAR_NAMES], dtype=dtype)
