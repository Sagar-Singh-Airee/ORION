from __future__ import annotations

import torch
import torch.nn as nn

from ...utils.registry import FUSION


@FUSION.register("gated_fusion")
class GatedFusion(nn.Module):
    """Learn whether a report should influence a visual prediction."""
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 512, dropout: float = 0.1, **_: object):
        super().__init__()
        self.vision = nn.Linear(vision_dim, hidden_dim)
        self.text = nn.Linear(text_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, vision: torch.Tensor, text: torch.Tensor, **_: object) -> torch.Tensor:
        if vision.ndim == 3:
            vision = vision.mean(dim=1)
        if text.ndim == 3:
            text = text[:, 0]
        visual, report = self.vision(vision), self.text(text)
        gate = self.gate(torch.cat([visual, report], dim=-1))
        return self.dropout(gate * visual + (1 - gate) * report)
