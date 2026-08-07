from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    def __init__(self, losses: Sequence[tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleList(loss for loss, _ in losses)
        self.weights = [float(weight) for _, weight in losses]

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return sum(weight * loss(*args, **kwargs) for loss, weight in zip(self.losses, self.weights, strict=True))
