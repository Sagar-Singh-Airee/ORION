"""Backbone factory with one clear failure point for optional timm models."""
from __future__ import annotations

from typing import Any

from ...utils.registry import BACKBONES
from .resnet import ResNet
from .timm_wrapper import build_timm_backbone


def build_backbone(name: str, pretrained: bool = True, in_channels: int = 1, **kwargs: Any):
    if name in {"resnet50_scratch", "resnet50_from_scratch"}:
        return ResNet(in_channels=in_channels)
    if name == "resnet50" and not pretrained:
        return ResNet(in_channels=in_channels)
    if name in BACKBONES:
        return BACKBONES.build(name, pretrained=pretrained, in_channels=in_channels, **kwargs)
    return build_timm_backbone(name, pretrained=pretrained, in_channels=in_channels, **kwargs)
