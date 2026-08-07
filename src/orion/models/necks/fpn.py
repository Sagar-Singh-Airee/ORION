"""Small feature-pyramid adapter for CNN feature maps."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int):
        super().__init__(); self.lateral = nn.ModuleList(nn.Conv2d(channel, out_channels, 1) for channel in in_channels); self.output = nn.ModuleList(nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels)
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.lateral): raise ValueError("Feature count mismatch")
        results = [None] * len(features); top = None
        for index in range(len(features) - 1, -1, -1):
            current = self.lateral[index](features[index])
            if top is not None: current = current + F.interpolate(top, size=current.shape[-2:], mode="nearest")
            results[index] = self.output[index](current); top = current
        return results  # type: ignore[return-value]
