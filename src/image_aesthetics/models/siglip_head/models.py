"""Small distributional aesthetic heads for cached SigLIP embeddings."""
from __future__ import annotations
import torch
from torch import nn

class DistributionHead(nn.Module):
    def __init__(self, layers: list[int], dropouts: list[float] | None = None):
        super().__init__()
        modules: list[nn.Module] = [nn.LayerNorm(layers[0])]
        dropouts = dropouts or [0.0] * (len(layers)-2)
        for a, b in zip(layers, layers[1:]):
            modules.append(nn.Linear(a, b))
            if b != 10:
                modules.append(nn.GELU())
                d = dropouts[len([x for x in modules if isinstance(x, nn.Linear)])-1]
                if d: modules.append(nn.Dropout(d))
        self.net = nn.Sequential(*modules)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

def build_head(name: str, input_dim: int = 768) -> DistributionHead:
    specs = {"A0": ([input_dim, 10], []), "A1": ([input_dim, 256, 10], [.10]),
             "A2": ([input_dim, 512, 128, 10], [.20, .10]), "A3": ([input_dim, 256, 128, 64, 10], [.20, .10, .10])}
    if name not in specs: raise ValueError(f"unknown architecture {name}")
    model = DistributionHead(*specs[name])
    if model.param_count >= 1_000_000: raise ValueError("head exceeds parameter budget")
    return model
