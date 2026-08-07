"""
Attention Pooling (Multi-Instance Learning Aggregator)

WHY it exists:
Standard average pooling treats all pixels/slices equally. But in knee MRI,
a meniscus tear might only be visible in 2 out of 30 slices, and in a tiny 10x10 pixel area.
Average pooling dilutes that signal, causing false negatives.

Attention Pooling learns to assign a weight to each feature vector.
It acts as a trainable weighted average, allowing the network to say:
"Ignore the first 10 slices, focus entirely on slice 15, specifically the medial side."

This is the core of Multi-Instance Learning (MIL) as proposed by Ilse et al. (2018).
"""

import torch
import torch.nn as nn
from ...utils.registry import NECKS

@NECKS.register("attention_pool")
class AttentionPool(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Attention mechanism (Gated Attention from Ilse et al.)
        # WHY Gated? It helps learn more complex, non-linear relationships compared to simple V^T * tanh(W*H)
        self.attention_v = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input features. Shape can be:
               - [B, Slices, C, H, W] -> Spatial slice aggregation
               - [B, Tokens, C] -> ViT token aggregation
               - [B, N, C] -> General MIL bag aggregation
            mask: Optional boolean mask [B, N] (True for valid, False for padded)
            
        Returns:
            Aggregated features [B, C]
        """
        # 1. Flatten spatial dimensions if present
        # If input is [B, Slices, C, H, W] -> reshape to [B, Slices*H*W, C]
        if x.dim() == 5:
            B, S, C, H, W = x.shape
            x = x.permute(0, 1, 3, 4, 2).reshape(B, S * H * W, C)
        elif x.dim() == 4:
            # e.g., [B, C, H, W] single image -> [B, H*W, C]
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).permute(0, 2, 1)
        elif x.dim() == 3:
            # Already [B, N, C]
            pass
        else:
            raise ValueError(f"Unsupported input shape for AttentionPool: {x.shape}")
            
        B, N, C = x.shape
        
        # 2. Compute attention scores
        # A_v = tanh(V * h^T), A_u = sigmoid(U * h^T)
        a_v = self.attention_v(x)  # [B, N, hidden_dim]
        a_u = self.attention_u(x)  # [B, N, hidden_dim]
        
        # Element-wise multiplication (the "Gate")
        gated = a_v * a_u
        
        # Final weights mapping to scalar
        scores = self.attention_weights(gated)  # [B, N, 1]
        scores = scores.squeeze(-1)             # [B, N]
        
        # 3. Apply mask if provided
        if mask is not None:
            # For 5D input, we need to expand the slice mask [B, S] to [B, S*H*W]
            if mask.size(1) != N:
                if "H" not in locals():
                    raise ValueError("Mask length must match token count for non-spatial input")
                mask = mask.unsqueeze(-1).expand(-1, -1, H * W).reshape(B, -1)
                
            # Set padded elements to a very large negative number so softmax -> 0
            mask = mask.to(dtype=torch.bool, device=scores.device)
            if not mask.any(dim=1).all():
                raise ValueError("Every bag must contain at least one valid token")
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            
        # 4. Softmax to get probabilities (weights sum to 1)
        weights = torch.softmax(scores, dim=-1)  # [B, N]
        
        # 5. Weighted average
        # x: [B, N, C], weights: [B, N]
        # output = sum_N (weights * x) -> [B, C]
        weights = weights.unsqueeze(-1)  # [B, N, 1]
        out = torch.sum(x * weights, dim=1)  # [B, C]
        
        return out
