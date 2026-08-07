from __future__ import annotations

import torch
import torch.nn as nn


class MultiViewAttention(nn.Module):
    """Aggregate sequence/view features [B, V, C] with a learned attention score."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(feature_dim, feature_dim // 2), nn.Tanh(), nn.Linear(feature_dim // 2, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.score(features).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        return torch.einsum("bv,bvc->bc", scores.softmax(-1), features)
