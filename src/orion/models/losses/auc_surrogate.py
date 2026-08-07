"""Pairwise logistic ranking loss, a smooth ROC-AUC surrogate."""
from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseAUCLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        terms = []
        for label in range(logits.shape[1]):
            known = targets[:, label] >= 0
            positive = logits[known & (targets[:, label] > 0.5), label]
            negative = logits[known & (targets[:, label] <= 0.5), label]
            if len(positive) and len(negative): terms.append(torch.nn.functional.softplus(-(positive[:, None] - negative[None, :])).mean())
        return torch.stack(terms).mean() if terms else logits.sum() * 0
