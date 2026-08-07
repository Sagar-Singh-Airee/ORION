"""ConvNeXt factory."""
from __future__ import annotations
from .timm_wrapper import build_timm_backbone

def convnext_large(pretrained: bool = True, **kwargs): return build_timm_backbone("convnext_large", pretrained=pretrained, **kwargs)
