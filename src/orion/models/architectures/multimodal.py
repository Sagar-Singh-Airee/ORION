"""
Full Multimodal Architecture (End-to-End)

WHY it exists:
This is the master class that wires together the backbone, neck, text encoder,
fusion module, and classification head based on the YAML configuration.
It abstracts away the complexity so the Trainer just sees a single `model(image, text)`.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...utils.registry import BACKBONES, NECKS, FUSION, HEADS
from ..text_encoders.registry import TEXT_ENCODERS  # Assuming a similar registry exists
from loguru import logger

class ORIONMultimodalModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        
        # 1. Vision Backbone
        v_cfg = config.model.vision_backbone
        self.vision_backbone = BACKBONES.build(v_cfg.name, pretrained=v_cfg.pretrained)
        v_dim = getattr(self.vision_backbone, "out_channels", 2048) # Default to ResNet50 dim
        
        # 2. Vision Neck (e.g., Attention Pooling for MIL)
        n_cfg = config.model.neck
        self.vision_neck = NECKS.build(n_cfg.type, in_dim=v_dim, hidden_dim=n_cfg.hidden_dim)
        v_pooled_dim = n_cfg.hidden_dim
        
        # 3. Text Encoder (Optional)
        self.is_multimodal = config.model.get("text_encoder") is not None
        if self.is_multimodal:
            t_cfg = config.model.text_encoder
            # Dummy fallback if registry not fully implemented yet
            if TEXT_ENCODERS.contains(t_cfg.name):
                self.text_encoder = TEXT_ENCODERS.build(t_cfg.name, pretrained=t_cfg.pretrained)
            else:
                logger.warning(f"Text encoder {t_cfg.name} not in registry. Using Identity.")
                self.text_encoder = nn.Identity()
                
            t_dim = getattr(self.text_encoder, "out_channels", 768) # Default RoBERTa dim
            
            # 4. Fusion Module
            f_cfg = config.model.fusion
            if f_cfg.type != "none":
                self.fusion = FUSION.build(f_cfg.type, vision_dim=v_pooled_dim, text_dim=t_dim, hidden_dim=f_cfg.hidden_dim)
                head_in_dim = f_cfg.hidden_dim
            else:
                # Early fusion fallback (concat)
                self.fusion = None
                head_in_dim = v_pooled_dim + t_dim
        else:
            self.fusion = None
            head_in_dim = v_pooled_dim
            
        # 5. Classification Head
        h_cfg = config.model.head
        self.head = HEADS.build(
            "multi_label", 
            in_dim=head_in_dim, 
            num_classes=config.labels.num_classes,
            hidden_dims=list(h_cfg.hidden_dims),
            dropout=h_cfg.dropout
        )

    def forward(self, image: torch.Tensor, text_inputs: dict | None = None, slice_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            image: [B, Slices, C, H, W] for MIL, or [B, C, H, W] for 2D
            text_inputs: dict with input_ids and attention_mask
            slice_mask: [B, Slices] boolean mask
        """
        # Vision Path
        v_feat = self.vision_backbone(image)
        v_pooled = self.vision_neck(v_feat, mask=slice_mask)
        
        # Text Path & Fusion
        if self.is_multimodal and text_inputs is not None:
            t_feat = self.text_encoder(**text_inputs)
            
            if self.fusion is not None:
                # Pass mask if using cross attention
                t_mask = text_inputs.get("attention_mask")
                fused_feat = self.fusion(v_pooled, t_feat, vision_mask=None, text_mask=t_mask)
            else:
                # Simple concat
                t_pooled = t_feat.mean(dim=1) if t_feat.dim() == 3 else t_feat
                fused_feat = torch.cat([v_pooled, t_pooled], dim=-1)
        else:
            fused_feat = v_pooled
            
        # Classification
        logits = self.head(fused_feat)
        return logits
