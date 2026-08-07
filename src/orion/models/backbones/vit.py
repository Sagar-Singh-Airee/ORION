"""ViT factory."""
from __future__ import annotations
from .timm_wrapper import build_timm_backbone

def vit_base_patch16(pretrained: bool = True, **kwargs): return build_timm_backbone("vit_base_patch16_224", pretrained=pretrained, **kwargs)
