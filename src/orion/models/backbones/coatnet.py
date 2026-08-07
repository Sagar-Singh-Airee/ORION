"""CoAtNet factory when a compatible timm model is installed."""
from __future__ import annotations
from .timm_wrapper import build_timm_backbone

def coatnet_0(pretrained: bool = True, **kwargs): return build_timm_backbone("coatnet_0", pretrained=pretrained, **kwargs)
