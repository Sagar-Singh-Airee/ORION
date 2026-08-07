"""Attention MIL pooling over variable-length MRI slice bags."""
from __future__ import annotations

import torch
import torch.nn as nn

from ...utils.registry import NECKS


@NECKS.register("slice_attention")
class SliceAttentionAggregator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(64, in_dim // 2)
        self.score = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.out_channels = in_dim

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None, return_attention: bool = False):
        if features.ndim != 3:
            raise ValueError(f"Expected [B, S, C] features, got {tuple(features.shape)}")
        scores = self.score(features).squeeze(-1)
        if mask is not None:
            mask = mask.to(dtype=torch.bool, device=scores.device)
            if mask.shape != scores.shape or not mask.any(dim=1).all():
                raise ValueError("mask must be [B, S] with at least one valid slice per study")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bs,bsc->bc", weights, features)
        return (pooled, weights) if return_attention else pooled


@NECKS.register("mean_pool")
class MaskedMeanAggregator(nn.Module):
    def __init__(self, in_dim: int, **_: object):
        super().__init__()
        self.out_channels = in_dim

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return features.mean(dim=1)
        weights = mask.to(features.dtype).unsqueeze(-1)
        return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
