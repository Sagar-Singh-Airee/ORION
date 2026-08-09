"""Image-only multi-instance MRI classifier.

Encodes every slice in a study with a shared backbone, aggregates only the
*valid* slices with attention pooling, and classifies at the study level.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..backbones import build_backbone
from ..heads.classification import MultiLabelHead
from ..necks.slice_aggregator import SliceAttentionAggregator

__all__ = ["ImageOnlyMILModel"]

# Attribute names different backbone families use to expose their output width.
# Checked in order so any backbone in the registry (CNN, ViT, timm wrapper, ...)
# is supported without special-casing each one here.
_FEATURE_DIM_ATTRS = ("out_channels", "num_features", "feature_dim", "embed_dim")


def _resolve_feature_dim(backbone: nn.Module) -> int:
    """Find the backbone's output feature width, or fail loudly if none is exposed."""
    for attr in _FEATURE_DIM_ATTRS:
        value = getattr(backbone, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise AttributeError(
        f"{backbone.__class__.__name__} does not expose a usable feature "
        f"dimension. Expected one of {_FEATURE_DIM_ATTRS} to be a positive int; "
        "add one of these attributes to the backbone before using it in ImageOnlyMILModel."
    )


def _pool_backbone_features(features: torch.Tensor) -> torch.Tensor:
    """Collapse a per-slice backbone output down to a single feature vector.

    Accepts, per slice:
      [N, C]       already pooled                    -> returned as-is
      [N, T, C]    token sequence (ViT-style)         -> mean-pooled over tokens
      [N, C, H, W] spatial feature map (CNN-style)    -> global-average-pooled
    """
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features.mean(dim=1)
    if features.ndim == 4:
        return features.mean(dim=(-2, -1))
    raise ValueError(
        "Backbone output must be 2-D [N,C], 3-D [N,T,C], or 4-D [N,C,H,W]; "
        f"got {tuple(features.shape)}"
    )


class ImageOnlyMILModel(nn.Module):
    """Vision-only multi-instance-learning pipeline: slices -> study prediction.

    Accepts the same call signature as the multimodal/text-only architectures
    (``image``, ``text_inputs``, ``slice_mask``) so it is interchangeable with
    them inside a shared training loop; ``text_inputs`` is accepted but unused.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50_scratch",
        num_classes: int = 12,
        pretrained: bool = False,
        in_channels: int = 1,
        head_hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        **backbone_kwargs: object,
    ) -> None:
        super().__init__()
        self.backbone = build_backbone(
            backbone_name, pretrained=pretrained, in_channels=in_channels, **backbone_kwargs
        )
        self.feature_dim = _resolve_feature_dim(self.backbone)
        self.aggregator = SliceAttentionAggregator(self.feature_dim, dropout=dropout)
        self.head = MultiLabelHead(
            self.feature_dim, num_classes, head_hidden_dims or [self.feature_dim // 2], dropout
        )

    def _validate_slice_mask(self, slice_mask: torch.Tensor, batch: int, slices: int) -> torch.Tensor:
        if slice_mask.shape != (batch, slices):
            raise ValueError(
                f"slice_mask must have shape [B, S] = {(batch, slices)}, "
                f"got {tuple(slice_mask.shape)}"
            )
        return slice_mask.bool()

    def encode_slices(self, image: torch.Tensor, slice_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode each slice of each study into a feature vector.

        Args:
            image: [B, S, C, H, W] stack of slices per study.
            slice_mask: optional [B, S] bool/float mask; ``True``/1 marks a real
                slice, ``False``/0 marks padding. When given, the backbone only
                runs on real slices — padding never touches the network.

        Returns:
            [B, S, feature_dim] per-slice features (zeros for padded slices).
        """
        if image.ndim != 5:
            raise ValueError(f"Expected image [B, S, C, H, W], got {tuple(image.shape)}")
        batch, slices, channels, height, width = image.shape
        flat_image = image.reshape(batch * slices, channels, height, width)

        if slice_mask is None:
            features = _pool_backbone_features(self.backbone(flat_image))
            return features.reshape(batch, slices, -1)

        flat_mask = self._validate_slice_mask(slice_mask, batch, slices).reshape(batch * slices)
        features = flat_image.new_zeros(batch * slices, self.feature_dim)
        if flat_mask.any():
            valid_features = _pool_backbone_features(self.backbone(flat_image[flat_mask]))
            features[flat_mask] = valid_features.to(features.dtype)
        return features.reshape(batch, slices, -1)

    def forward(
        self,
        image: torch.Tensor,
        text_inputs: dict | None = None,
        slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-class logits of shape [B, num_classes]."""
        del text_inputs  # unused: kept only for interface parity with other architectures
        slice_features = self.encode_slices(image, slice_mask)
        study_embedding = self.aggregator(slice_features, slice_mask)
        return self.head(study_embedding)

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}"