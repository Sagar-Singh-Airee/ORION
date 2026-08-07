"""Minimal classifier on top of a sequence text encoder."""
from __future__ import annotations

import torch
import torch.nn as nn


class TextOnlyModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int = 12):
        super().__init__()
        self.encoder, self.head = encoder, nn.Linear(feature_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if features.ndim == 3:
            features = features[:, 0]
        return self.head(features)
