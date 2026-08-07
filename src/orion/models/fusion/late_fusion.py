from __future__ import annotations

import torch
import torch.nn as nn

from ...utils.registry import FUSION


@FUSION.register("late_fusion")
class LateFusion(nn.Module):
    """Blend two aligned logits; useful for independently trained modalities."""
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int | None = None, **_: object):
        super().__init__()
        if vision_dim != text_dim:
            raise ValueError("LateFusion requires equally sized logit vectors")
        self.logit_gate = nn.Parameter(torch.zeros(vision_dim))

    def forward(self, vision: torch.Tensor, text: torch.Tensor, **_: object) -> torch.Tensor:
        return torch.sigmoid(self.logit_gate) * vision + (1 - torch.sigmoid(self.logit_gate)) * text
