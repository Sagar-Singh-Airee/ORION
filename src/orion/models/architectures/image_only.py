"""Image-only multi-instance MRI classifier."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..backbones import build_backbone
from ..heads.classification import MultiLabelHead
from ..necks.slice_aggregator import SliceAttentionAggregator


class ImageOnlyMILModel(nn.Module):
    """Encode every slice, aggregate only valid slices, then classify the study."""
    def __init__(self, backbone_name: str = "resnet50_scratch", num_classes: int = 12, pretrained: bool = False, in_channels: int = 1, head_hidden_dims: list[int] | None = None, dropout: float = 0.3, **backbone_kwargs: object):
        super().__init__()
        self.backbone = build_backbone(backbone_name, pretrained=pretrained, in_channels=in_channels, **backbone_kwargs)
        self.feature_dim = int(getattr(self.backbone, "out_channels"))
        self.aggregator = SliceAttentionAggregator(self.feature_dim, dropout=dropout)
        self.head = MultiLabelHead(self.feature_dim, num_classes, head_hidden_dims or [self.feature_dim // 2], dropout)

    def encode_slices(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"Expected image [B, S, C, H, W], got {tuple(image.shape)}")
        batch, slices, channels, height, width = image.shape
        features = self.backbone(image.reshape(batch * slices, channels, height, width))
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim == 3:
            features = features.mean(dim=1)
        if features.ndim != 2:
            raise ValueError(f"Backbone must produce 2-D, 3-D, or 4-D features, got {tuple(features.shape)}")
        return features.reshape(batch, slices, -1)

    def forward(self, image: torch.Tensor, text_inputs: dict | None = None, slice_mask: torch.Tensor | None = None) -> torch.Tensor:
        del text_inputs
        return self.head(self.aggregator(self.encode_slices(image), slice_mask))
