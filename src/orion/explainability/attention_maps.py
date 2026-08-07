"""Convert attention weights to normalized per-slice relevance scores."""
from __future__ import annotations

import torch


def slice_attention_scores(weights: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if weights.ndim != 2: raise ValueError("weights must be [B, S]")
    scores = weights.detach().float().clone()
    if mask is not None: scores = scores.masked_fill(~mask.bool(), 0)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(torch.finfo(scores.dtype).eps)
