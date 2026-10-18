"""Branch-aware residual fusion candidates for the ten-signal contract."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .contract import FEATURE_DIM, SCALAR_DIM, split_features


class F5ResidualBase(nn.Module):
    """Shared F2-logit base, bounded residual, and optional technical head.

    ``forward`` has a production-safe, query-independent signature and returns
    only the aesthetic score.  Dataset IDs and training-only calibration are
    handled by the trainer outside this module.
    """

    architecture_id = "F5-base-v1"
    residual_hidden: tuple[int, ...] = ()
    representation_dim = 0

    def __init__(self, *, technical_head: bool = False, residual_bound: float = 0.25):
        super().__init__()
        if residual_bound <= 0:
            raise ValueError("residual_bound must be positive")
        self.residual_bound = float(residual_bound)
        self.scalar_base = nn.Linear(SCALAR_DIM, 1)
        self.residual_head = self._make_residual_head()
        self._technical_head_enabled = bool(technical_head)
        self.technical_head = None
        self.f2_initialization_loaded = False
        self._reset_residual()

    def _make_residual_head(self) -> nn.Module:
        raise NotImplementedError

    def _represent(self, siglip: torch.Tensor, clip: torch.Tensor,
                   scalars: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _reset_residual(self) -> None:
        # Begin at the fold-specific F2 logit.  The final zero initialization
        # also makes the initial residual exactly zero while retaining a bounded
        # tanh correction as training proceeds.
        final = self.residual_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        if getattr(self, "gate", None) is not None:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def _initialize_technical_head(self) -> None:
        if self._technical_head_enabled:
            self.technical_head = nn.Linear(self.representation_dim, 1)

    def initialize_from_f2(self, state_dict: dict[str, Any]) -> None:
        """Copy an F2 ``RegularizedLinear`` state into the shared scalar logit."""
        weight = state_dict.get("linear.weight")
        bias = state_dict.get("linear.bias")
        if not torch.is_tensor(weight) or tuple(weight.shape) != (1, SCALAR_DIM):
            raise ValueError("F2 initialization requires linear.weight with shape (1, 8)")
        if not torch.is_tensor(bias) or tuple(bias.shape) != (1,):
            raise ValueError("F2 initialization requires linear.bias with shape (1,)")
        with torch.no_grad():
            self.scalar_base.weight.copy_(weight.to(self.scalar_base.weight))
            self.scalar_base.bias.copy_(bias.to(self.scalar_base.bias))
        self.f2_initialization_loaded = True

    def architecture_config(self) -> dict[str, Any]:
        return {
            "model_id": self.architecture_id,
            "feature_contract": "joined-ten-v1",
            "feature_dimension": FEATURE_DIM,
            "scalar_order": ["a1_siglip", "laion_clip", "nima", "musiq_aesthetic",
                             "musiq_technical", "sharpness", "contrast", "colorfulness"],
            "residual_bound": self.residual_bound,
            "scalar_base": "fold-specific F2 pre-sigmoid logit",
            "technical_head": self.technical_head is not None,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }

    @property
    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def bounded_residual(self, features: torch.Tensor) -> torch.Tensor:
        siglip, clip, scalars = split_features(features)
        representation = self._represent(siglip, clip, scalars)
        raw = self.residual_head(representation).squeeze(-1)
        return self.residual_bound * torch.tanh(raw)

    def _score_and_representation(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        siglip, clip, scalars = split_features(features)
        representation = self._represent(siglip, clip, scalars)
        # The optional technical head taps the penultimate residual feature.
        # This follows the memo's 32/128/64-wide head contracts for R/G/X.
        # It is not a replacement score path and is only surfaced explicitly
        # through ``forward_with_aux``.
        residual_hidden = self.residual_head[:-1](representation)
        residual_logit = self.residual_head[-1](residual_hidden).squeeze(-1)
        residual = self.residual_bound * torch.tanh(residual_logit)
        logit = self.scalar_base(scalars).squeeze(-1) + residual
        return torch.sigmoid(logit), residual_hidden

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return aesthetic scores in [0, 1], without dataset IDs or query data."""
        return self._score_and_representation(features)[0]

    def forward_with_aux(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q_aes, q_tech)`` for an explicitly enabled auxiliary task."""
        if self.technical_head is None:
            raise RuntimeError("technical head is disabled for this model")
        aesthetic, representation = self._score_and_representation(features)
        technical = torch.sigmoid(self.technical_head(representation).squeeze(-1))
        return aesthetic, technical


class F5R(F5ResidualBase):
    """F5-R: additive embedding residual over the F2 scalar score."""

    architecture_id = "F5-R-v1"
    representation_dim = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.siglip_branch = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 96), nn.GELU(), nn.Linear(96, 32))
        self.clip_branch = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 96), nn.GELU(), nn.Linear(96, 32))
        self._initialize_technical_head()

    def _make_residual_head(self):
        return nn.Sequential(nn.Linear(72, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))

    def _represent(self, siglip, clip, scalars):
        return torch.cat((self.siglip_branch(siglip), self.clip_branch(clip), scalars), dim=-1)


class F5G(F5ResidualBase):
    """F5-G: bounded content gates with a protected raw-scalar residual input."""

    architecture_id = "F5-G-v1"
    representation_dim = 128

    def __init__(self, *, gate_enabled: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.gate_enabled = bool(gate_enabled)
        self.siglip_branch = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64))
        self.clip_branch = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64))
        self.gate = (nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, SCALAR_DIM))
                     if self.gate_enabled else None)
        self.scalar_branch = nn.Sequential(nn.Linear(SCALAR_DIM, 32), nn.GELU())
        if self.gate is not None:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        self._initialize_technical_head()

    @property
    def architecture_id(self):
        return "F5-G-v1" if self.gate_enabled else "F5-G-NoGate-v1"

    def architecture_config(self):
        config = super().architecture_config()
        config["content_gate"] = self.gate_enabled
        config["gate_parameter_delta_vs_F5-G"] = 0 if self.gate_enabled else -4392
        return config

    def _make_residual_head(self):
        return nn.Sequential(nn.Linear(168, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def _represent(self, siglip, clip, scalars):
        siglip_hidden = self.siglip_branch(siglip)
        clip_hidden = self.clip_branch(clip)
        context = torch.cat((siglip_hidden, clip_hidden), dim=-1)
        gated_scalars = (scalars * (1.0 + 0.25 * torch.tanh(self.gate(context)))
                         if self.gate is not None else scalars)
        scalar_hidden = self.scalar_branch(gated_scalars)
        return torch.cat((siglip_hidden, clip_hidden, scalar_hidden, scalars), dim=-1)

    def gate_values(self, features: torch.Tensor) -> torch.Tensor:
        if self.gate is None:
            return torch.ones_like(split_features(features)[2])
        siglip, clip, _ = split_features(features)
        context = torch.cat((self.siglip_branch(siglip), self.clip_branch(clip)), dim=-1)
        return 1.0 + 0.25 * torch.tanh(self.gate(context))


class F5X(F5ResidualBase):
    """F5-X: rank-16 embedding-by-scalar interaction residual."""

    architecture_id = "F5-X-v1"
    representation_dim = 64

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.siglip_branch = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64))
        self.clip_branch = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64))
        self.context_projection = nn.Linear(128, 16)
        self.scalar_projection = nn.Linear(SCALAR_DIM, 16)
        self._initialize_technical_head()

    def _make_residual_head(self):
        return nn.Sequential(nn.Linear(152, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def _represent(self, siglip, clip, scalars):
        siglip_hidden = self.siglip_branch(siglip)
        clip_hidden = self.clip_branch(clip)
        context = self.context_projection(torch.cat((siglip_hidden, clip_hidden), dim=-1))
        scalar = self.scalar_projection(scalars)
        return torch.cat((siglip_hidden, clip_hidden, context * scalar, scalars), dim=-1)
