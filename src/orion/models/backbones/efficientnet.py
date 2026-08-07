"""EfficientNet factory."""
from __future__ import annotations
from .timm_wrapper import build_timm_backbone

def efficientnet_b4(pretrained: bool = True, **kwargs): return build_timm_backbone("efficientnet_b4", pretrained=pretrained, **kwargs)
