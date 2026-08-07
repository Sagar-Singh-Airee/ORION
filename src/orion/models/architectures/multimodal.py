"""Vision + report multi-instance model with optional modality fusion."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from ...data.text.label_extractor import FINDINGS
from ...utils.registry import FUSION
from ..backbones import build_backbone
from ..heads.classification import MultiLabelHead
from ..necks.slice_aggregator import SliceAttentionAggregator
from ..text_encoders import TEXT_ENCODERS


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping): return value.get(key, default)
    return getattr(value, key, default)


class ORIONMultimodalModel(nn.Module):
    """A valid image-only model when text input is unavailable at inference time."""
    def __init__(self, config: Any):
        super().__init__()
        model_cfg = _get(config, "model", config)
        vision_cfg = _get(model_cfg, "vision_backbone", _get(model_cfg, "backbone", {}))
        backbone_name = _get(vision_cfg, "name", "resnet50_scratch")
        self.vision_backbone = build_backbone(backbone_name, pretrained=bool(_get(vision_cfg, "pretrained", False)), in_channels=int(_get(vision_cfg, "in_channels", 1)), drop_path_rate=float(_get(vision_cfg, "drop_path_rate", 0.0)))
        feature_dim = int(self.vision_backbone.out_channels)
        neck_cfg = _get(model_cfg, "neck", {})
        self.vision_neck = SliceAttentionAggregator(feature_dim, hidden_dim=_get(neck_cfg, "hidden_dim", None), dropout=float(_get(neck_cfg, "dropout", 0.0)))
        text_cfg = _get(model_cfg, "text_encoder", None)
        self.text_encoder = None
        self.fusion = None
        output_dim = feature_dim
        data_cfg = _get(config, "data", {})
        if text_cfg and bool(_get(data_cfg, "multimodal", False) or _get(text_cfg, "enabled", False)):
            # Import registrations before the registry lookup.
            import orion.models.text_encoders  # noqa: F401
            self.text_encoder = TEXT_ENCODERS.build(_get(text_cfg, "name", "xlm_roberta_base"), pretrained=bool(_get(text_cfg, "pretrained", True)), dropout=float(_get(text_cfg, "dropout", 0.0)))
            fusion_cfg = _get(model_cfg, "fusion", {"type": "gated_fusion"})
            fusion_type = _get(fusion_cfg, "type", _get(fusion_cfg, "name", "gated_fusion"))
            if fusion_type != "none":
                import orion.models.fusion  # noqa: F401
                hidden = int(_get(fusion_cfg, "hidden_dim", feature_dim))
                self.fusion = FUSION.build(fusion_type, vision_dim=feature_dim, text_dim=self.text_encoder.out_channels, hidden_dim=hidden, num_heads=int(_get(fusion_cfg, "num_heads", 8)), dropout=float(_get(fusion_cfg, "dropout", 0.1)))
                output_dim = hidden
        head_cfg = _get(model_cfg, "head", {})
        labels_cfg = _get(config, "labels", {})
        self.head = MultiLabelHead(output_dim, int(_get(labels_cfg, "num_classes", _get(model_cfg, "num_classes", len(FINDINGS)))), list(_get(head_cfg, "hidden_dims", [output_dim // 2])), float(_get(head_cfg, "dropout", 0.3)), _get(head_cfg, "activation", "gelu"))

    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5: raise ValueError("image must be [B, S, C, H, W]")
        batch, slices, channels, height, width = image.shape
        features = self.vision_backbone(image.reshape(batch * slices, channels, height, width))
        if features.ndim == 4: features = features.mean(dim=(-2, -1))
        elif features.ndim == 3: features = features.mean(dim=1)
        return features.reshape(batch, slices, -1)

    def forward(self, image: torch.Tensor, text_inputs: dict[str, torch.Tensor] | None = None, slice_mask: torch.Tensor | None = None) -> torch.Tensor:
        visual = self.vision_neck(self._encode_image(image), slice_mask)
        fused = visual
        if self.text_encoder is not None and self.fusion is not None and text_inputs is not None:
            text = self.text_encoder(**text_inputs)
            fused = self.fusion(visual, text, text_mask=text_inputs.get("attention_mask"))
        return self.head(fused)
