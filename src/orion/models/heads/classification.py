"""
Multi-Label Classification Head

WHY it exists:
Maps the high-dimensional feature vectors from the backbone/fusion layer down
to the 12 target classes. 
"""

import torch
import torch.nn as nn
from typing import List
from ...utils.registry import HEADS

@HEADS.register("multi_label")
class MultiLabelHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 12, hidden_dims: List[int] = [256], 
                 dropout: float = 0.3, activation: str = "gelu"):
        super().__init__()
        
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        
        layers = []
        current_dim = in_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                act_fn,
                nn.Dropout(dropout)
            ])
            current_dim = h_dim
            
        # Final classification layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns RAW LOGITS (no sigmoid applied).
        Loss functions (BCEWithLogitsLoss, AsymmetricLoss) expect logits for numeric stability.
        """
        return self.net(x)
