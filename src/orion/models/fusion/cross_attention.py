"""
Cross-Attention Fusion

WHY it exists:
Simple concatenation (Early Fusion) doesn't allow one modality to influence
how features are extracted from the other.
Cross-Attention allows the Vision model to "query" the Text report, finding
exactly which words support a visual finding, and vice versa.
"""

import torch
import torch.nn as nn
from ...utils.registry import FUSION

@FUSION.register("cross_attention")
class CrossAttentionFusion(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Project inputs to a common hidden space
        self.proj_v = nn.Linear(vision_dim, hidden_dim) if vision_dim != hidden_dim else nn.Identity()
        self.proj_t = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()
        
        # Multi-Head Attention: Vision queries Text
        # "Given this visual feature, what part of the report is relevant?"
        self.v_queries_t = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Multi-Head Attention: Text queries Vision
        # "Given this report text, where is the corresponding visual evidence?"
        self.t_queries_v = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # Final projection to combine them
        # Concatenate the two cross-attended features and project back to hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Layer Norms for stability
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor, vision_mask: torch.Tensor | None = None, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            vision_feat: [B, N_v, C_v]
            text_feat: [B, N_t, C_t]
            vision_mask: [B, N_v] (True=Valid, False=Pad)
            text_mask: [B, N_t]
            
        Returns:
            Fused representation [B, hidden_dim]
        """
        # Ensure 3D (Sequence length = 1 if pooled features are passed)
        if vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(1)
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
            
        # Project to common space
        v = self.proj_v(vision_feat) # [B, N_v, H]
        t = self.proj_t(text_feat)   # [B, N_t, H]
        
        # Cross Attention
        # Note: PyTorch MultiheadAttention expects key_padding_mask to be True for PADDED elements
        # If our mask is True for VALID elements, we must invert it (~).
        
        v_pad_mask = ~vision_mask if vision_mask is not None else None
        t_pad_mask = ~text_mask if text_mask is not None else None
        
        # Vision queries Text: Q=v, K=t, V=t
        attn_out_v, _ = self.v_queries_t(query=v, key=t, value=t, key_padding_mask=t_pad_mask)
        v_attended = self.norm_v(v + attn_out_v)
        
        # Text queries Vision: Q=t, K=v, V=v
        attn_out_t, _ = self.t_queries_v(query=t, key=v, value=v, key_padding_mask=v_pad_mask)
        t_attended = self.norm_t(t + attn_out_t)
        
        # Pool sequences if they aren't already pooled (simple average for now)
        # In a real model, you might use a CLS token or attention pooling here too
        v_pooled = v_attended.mean(dim=1) # [B, H]
        t_pooled = t_attended.mean(dim=1) # [B, H]
        
        # Concatenate and project
        fused = torch.cat([v_pooled, t_pooled], dim=-1) # [B, 2*H]
        out = self.fc(fused) # [B, H]
        
        return out
