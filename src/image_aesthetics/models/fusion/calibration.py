"""Training-only monotonic native-scale calibration heads."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class MonotonicAffineCalibrator(nn.Module):
    """Positive-slope affine map from shared q in [0,1] to native labels."""
    def __init__(self, native_min: float, native_max: float):
        super().__init__(); self.native_min=float(native_min); self.native_max=float(native_max)
        span=max(1e-6,self.native_max-self.native_min)
        self.raw_slope=nn.Parameter(torch.tensor(float(torch.log(torch.expm1(torch.tensor(span))))))
        self.bias=nn.Parameter(torch.tensor(self.native_min))
    @property
    def slope(self): return F.softplus(self.raw_slope)
    def forward(self, q): return self.bias + self.slope * q

class DatasetCalibrationHeads(nn.Module):
    """Collection of dataset-specific calibrators; dataset ID is never production input."""
    def __init__(self, ranges: dict[str, tuple[float,float]]):
        super().__init__(); self.dataset_ids=tuple(ranges); self.heads=nn.ModuleDict({k: MonotonicAffineCalibrator(*v) for k,v in ranges.items()})
    def forward(self, q: torch.Tensor, dataset_ids: list[str] | tuple[str,...]) -> torch.Tensor:
        if q.ndim != 1 or len(dataset_ids) != len(q): raise ValueError("q and dataset_ids must have matching batch length")
        out=torch.empty_like(q)
        for name in self.dataset_ids:
            mask=torch.tensor([x == name for x in dataset_ids], device=q.device)
            if mask.any(): out[mask]=self.heads[name](q[mask])
        unknown=set(dataset_ids)-set(self.dataset_ids)
        if unknown: raise ValueError(f"unknown datasets: {sorted(unknown)}")
        return out
