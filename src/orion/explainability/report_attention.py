"""Rank report tokens using cross-attention weights."""
from __future__ import annotations

import torch


def rank_tokens(tokens: list[str], attention: torch.Tensor, top_k: int = 10) -> list[tuple[str, float]]:
    if attention.shape[-1] != len(tokens): raise ValueError("Token count must match attention's final dimension")
    weights = attention.detach().float().mean(dim=tuple(range(attention.ndim - 1)))
    indices = weights.topk(min(top_k, len(tokens))).indices.tolist()
    return [(tokens[index], float(weights[index])) for index in indices]
