"""Swin V2 factory; implementation is delegated to maintained timm releases."""
from __future__ import annotations
from .timm_wrapper import build_timm_backbone

def swin_v2_base(pretrained: bool = True, **kwargs): return build_timm_backbone("swin_v2_base", pretrained=pretrained, **kwargs)
