"""Compatible fusion objectives."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .calibration import DatasetCalibrationHeads

def calibrated_huber_loss(q, native_target, dataset_ids, calibrators: DatasetCalibrationHeads, delta: float = 1.0):
    prediction=calibrators(q, dataset_ids)
    return F.huber_loss(prediction, native_target, delta=delta), prediction

def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor, *, groups=None, margin: float = 0.0, reduction: str = "mean"):
    if scores.shape != targets.shape: raise ValueError("scores and targets must have equal shape")
    if scores.ndim != 1: raise ValueError("pairwise loss expects a one-dimensional batch")
    diff=scores[:,None]-scores[None,:]; target_diff=targets[:,None]-targets[None,:]
    mask=target_diff.abs()>0
    if groups is not None:
        if len(groups) != len(scores): raise ValueError("groups must match the batch length")
        compatible=torch.tensor([[left == right for right in groups] for left in groups],device=scores.device,dtype=torch.bool)
        mask=mask & compatible
    if not mask.any(): return scores.sum()*0
    desired=torch.sign(target_diff[mask]); return F.softplus(-(desired*diff[mask]-margin)).mean() if reduction=="mean" else F.softplus(-(desired*diff[mask]-margin)).sum()

def technical_auxiliary_loss(prediction: torch.Tensor, target: torch.Tensor, *, valid: torch.Tensor | None = None):
    if valid is not None:
        if valid.dtype != torch.bool: valid=valid.bool()
        if not valid.any(): return prediction.sum()*0
        prediction,target=prediction[valid],target[valid]
    return F.huber_loss(prediction,target)
