"""Masked BCE for expert and abstaining weak labels."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.registry import LOSSES


@LOSSES.register("bce")
class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | list[float] | None = None, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32) if pos_weight is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        known = targets >= 0
        loss = F.binary_cross_entropy_with_logits(logits, targets.clamp(0, 1), pos_weight=self.pos_weight, reduction="none")
        loss = loss * known
        if self.reduction == "none": return loss
        if self.reduction == "sum": return loss.sum()
        return loss.sum() / known.sum().clamp_min(1)
