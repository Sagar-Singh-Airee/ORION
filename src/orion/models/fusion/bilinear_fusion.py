from __future__ import annotations

import torch
import torch.nn as nn

from ...utils.registry import FUSION


@FUSION.register("bilinear_fusion")
class BilinearFusion(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 512, dropout: float = 0.1, **_: object):
        super().__init__()
        self.vision, self.text = nn.Linear(vision_dim, hidden_dim), nn.Linear(text_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, vision: torch.Tensor, text: torch.Tensor, **_: object) -> torch.Tensor:
        if vision.ndim == 3: vision = vision.mean(dim=1)
        if text.ndim == 3: text = text[:, 0]
        return self.output(self.vision(vision) * self.text(text))
