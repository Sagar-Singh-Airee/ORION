from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("focal")
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        known = targets >= 0
        target = targets.clamp(0, 1)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        loss = (1 - pt).pow(self.gamma) * bce
        if self.alpha is not None: loss = loss * torch.where(target > 0, self.alpha, 1 - self.alpha)
        return (loss * known).sum() / known.sum().clamp_min(1)
