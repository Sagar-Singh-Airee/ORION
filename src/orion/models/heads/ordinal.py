"""Cumulative-link ordinal head for optional severity tasks."""
from __future__ import annotations

import torch
import torch.nn as nn


class OrdinalHead(nn.Module):
    def __init__(self, in_dim: int, num_levels: int):
        super().__init__()
        if num_levels < 2: raise ValueError("num_levels must be at least 2")
        self.score = nn.Linear(in_dim, 1); self.thresholds = nn.Parameter(torch.arange(num_levels - 1, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) - self.thresholds
