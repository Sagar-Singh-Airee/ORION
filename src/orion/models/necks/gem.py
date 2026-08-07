from __future__ import annotations

import torch
import torch.nn as nn


class GeM(nn.Module):
    """Generalized mean pooling; p=1 is average pooling, larger p approaches max."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))) if learnable else float(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp_min(1e-3) if isinstance(self.p, torch.Tensor) else self.p
        return x.clamp_min(self.eps).pow(p).mean(dim=(-2, -1)).pow(1.0 / p)
