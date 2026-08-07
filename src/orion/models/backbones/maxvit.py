"""MaxViT factory."""
from __future__ import annotations
from .timm_wrapper import build_timm_backbone

def maxvit_base(pretrained: bool = True, **kwargs): return build_timm_backbone("maxvit_base", pretrained=pretrained, **kwargs)
