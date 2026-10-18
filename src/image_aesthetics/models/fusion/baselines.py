"""F0-F4 fusion baselines with a stable inference API."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch
from torch import nn
from .contract import FEATURE_DIM, SCALAR_DIM, SCALAR_NAMES, split_features

# Query-independent portion of Moments Generator's fallback profile, renormalized
# after removing cosine.  Its `technical` composite is represented by equal
# sharpness/contrast/colorfulness contributions.
MOMENTS_F0_WEIGHTS = {
    "a1_siglip": 0.0, "laion_clip": 3/11, "nima": 3/11,
    "musiq_aesthetic": 3/11, "musiq_technical": 1/11,
    "sharpness": 1/33, "contrast": 1/33, "colorfulness": 1/33,
}

class WeightedScalarScore(nn.Module):
    """F0: configurable weighted sum of the eight normalized scalar signals."""
    def __init__(self, weights: Mapping[str, float] | Sequence[float] | None = None, bias: float = 0.0):
        super().__init__()
        if weights is None: weights = MOMENTS_F0_WEIGHTS
        if isinstance(weights, Mapping):
            missing = [n for n in SCALAR_NAMES if n not in weights]
            if missing: raise ValueError(f"missing weights: {missing}")
            values = [weights[n] for n in SCALAR_NAMES]
        else: values = list(weights)
        if len(values) != SCALAR_DIM: raise ValueError(f"expected {SCALAR_DIM} weights")
        self.register_buffer("weights", torch.as_tensor(values, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(float(bias), dtype=torch.float32))
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, _, scalars = split_features(features)
        return scalars @ self.weights + self.bias

class ScalarEnsemble(nn.Module):
    """F1: individual scalar heads and mean/median ensembles."""
    def __init__(self, mode: str = "mean", signal: str | None = None):
        super().__init__()
        if mode not in {"individual", "mean", "median"}: raise ValueError("mode must be individual, mean, or median")
        self.mode = mode
        self.signal = signal
        if mode == "individual" and signal not in SCALAR_NAMES:
            raise ValueError(f"individual mode requires signal in {SCALAR_NAMES}")
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, _, x = split_features(features)
        if self.mode == "individual": return x[..., SCALAR_NAMES.index(self.signal)]
        if self.mode == "mean": return x.mean(-1)
        ordered=x.sort(-1).values; mid=SCALAR_DIM // 2
        return (ordered[..., mid-1] + ordered[..., mid]) / 2

class RegularizedLinear(nn.Module):
    """F2: ridge-regularized linear model on scalar signals."""
    def __init__(self, l2: float = 1e-4, bias: bool = True):
        super().__init__(); self.linear = nn.Linear(SCALAR_DIM, 1, bias=bias); self.l2 = float(l2)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(split_features(features)[2]).squeeze(-1).sigmoid()
    def regularization_loss(self) -> torch.Tensor:
        return self.l2 * self.linear.weight.square().sum()

class ScalarMLP(nn.Module):
    """F3: scalar-only nonlinear baseline."""
    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(SCALAR_DIM), nn.Linear(SCALAR_DIM, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.net(split_features(features)[2]).squeeze(-1).sigmoid()

class CorrectedScalarMLP(nn.Module):
    """F3C: corrected F3 control without per-image scalar LayerNorm.

    Inputs are expected to have the existing fold-fitted normalization applied
    by the dataset view.  The historical ``ScalarMLP`` remains unchanged.
    """
    architecture_id = "F3C-no-per-image-layernorm-v1"

    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(SCALAR_DIM, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden // 2),
                                 nn.GELU(), nn.Linear(hidden // 2, 1))
        self.hidden = hidden
        self.dropout = float(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(split_features(features)[2]).squeeze(-1).sigmoid()

    def architecture_config(self) -> dict[str, object]:
        return {"model_id": self.architecture_id, "hidden": self.hidden,
                "dropout": self.dropout, "scalar_normalization": "fold-fitted only"}

class FlatFeatureMLP(nn.Module):
    """F4: flat 1,544-feature MLP kept below the one-million parameter budget."""
    def __init__(self, hidden: tuple[int, int] = (256, 128), dropout: float = 0.1):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, hidden[0]), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden[0], hidden[1]), nn.GELU(), nn.Linear(hidden[1], 1))
        if self.param_count >= 1_000_000: raise ValueError("F4 exceeds one-million parameter limit")
    @property
    def param_count(self): return sum(p.numel() for p in self.parameters())
    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.net(features).squeeze(-1).sigmoid()

def build_baseline(name: str, **kwargs) -> nn.Module:
    if name in {"F5-R", "F5-G", "F5-G-NoGate", "F5-X"}:
        from .f5 import F5G, F5R, F5X
        if name == "F5-G-NoGate":
            return F5G(gate_enabled=False, **kwargs)
        return {"F5-R": F5R, "F5-G": F5G, "F5-X": F5X}[name](**kwargs)
    factories = {"F0": WeightedScalarScore, "F1": ScalarEnsemble,
                 "F2": RegularizedLinear, "F3": ScalarMLP,
                 "F3C": CorrectedScalarMLP, "F4": FlatFeatureMLP}
    if name not in factories:
        raise ValueError(f"unknown fusion model {name!r}; choose from {sorted((*factories, 'F5-R', 'F5-G', 'F5-G-NoGate', 'F5-X'))}")
    return factories[name](**kwargs)
