from __future__ import annotations

import torch.nn as nn


class AuxiliaryHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__(); self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
    def forward(self, features): return self.net(features)
