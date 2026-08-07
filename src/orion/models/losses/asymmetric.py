"""
Asymmetric Loss for Multi-Label Classification

WHY it exists:
In medical datasets, especially multi-label ones like RSNA Knee, there is severe
positive/negative imbalance. An image might have 1 positive label and 11 negative labels.
Standard Binary Cross Entropy (BCE) treats all errors equally, so the network quickly
learns to predict "negative" for everything to get a low loss.

Focal Loss (Lin et al., 2017) down-weights easy examples.
Asymmetric Loss (Ben-Baruch et al., 2020) improves this specifically for multi-label:
It operates differently on positive and negative samples.
It completely discards extremely easy negatives (probability < clip_margin).
This forces the network to focus on finding the rare positive abnormalities.
"""

import torch
import torch.nn as nn
from ...utils.registry import LOSSES

@LOSSES.register("asymmetric")
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, 
                 clip: float = 0.05, eps: float = 1e-8, disable_torch_grad_focal_loss: bool = True):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples (higher = more focus on hard negatives).
            gamma_pos: Focusing parameter for positive samples (usually 0 or 1).
            clip: Probability margin. If a negative prediction is below this, loss is zeroed.
        """
        super().__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw logits (before sigmoid) [B, C]
            y: Binary targets [B, C]
        """
        known = y >= 0
        y = y.clamp(0, 1)
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        
        # Split probabilities into positive and negative classes
        # y is 1 for positive, 0 for negative
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        # If the network predicts a negative class with high confidence (p < clip),
        # we completely remove it from the loss calculation.
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Cross Entropy
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing (The "gamma" exponents)
        # Easy examples get down-weighted
        loss_pos = los_pos * torch.pow(1 - xs_pos, self.gamma_pos)
        loss_neg = los_neg * torch.pow(1 - xs_neg, self.gamma_neg)

        # Combine and average
        loss = loss_pos + loss_neg
        
        # Return negative sum (since log probabilities are negative)
        return -(loss * known).sum() / known.sum().clamp_min(1)
