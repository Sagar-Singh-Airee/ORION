from __future__ import annotations

import torch
import torch.nn as nn

from .bce import MaskedBCEWithLogitsLoss
from ...utils.registry import LOSSES


@LOSSES.register("label_smoothing_bce")
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        if not 0 <= smoothing < 0.5: raise ValueError("smoothing must be in [0, 0.5)")
        self.smoothing, self.bce = smoothing, MaskedBCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = torch.where(targets >= 0, targets * (1 - self.smoothing) + 0.5 * self.smoothing, targets)
        return self.bce(logits, smoothed)
