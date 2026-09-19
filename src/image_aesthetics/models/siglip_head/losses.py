"""Distributional AVA objectives."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def distribution_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = .25):
    pred = logits.softmax(-1)
    pred_cdf = pred.cumsum(-1); target_cdf = target.cumsum(-1)
    emd = ((pred_cdf - target_cdf) ** 2).mean()
    bins = torch.arange(1, 11, device=logits.device, dtype=logits.dtype)
    pm = (pred * bins).sum(-1); tm = (target * bins).sum(-1)
    huber = F.huber_loss((pm - 1) / 9, (tm - 1) / 9)
    return emd + alpha * huber, {"emd": float(emd.detach()), "huber": float(huber.detach())}
