"""Multimodal fusion modules."""
from .bilinear_fusion import BilinearFusion
from .cross_attention import CrossAttentionFusion
from .early_fusion import EarlyFusion
from .gated_fusion import GatedFusion

__all__ = ["BilinearFusion", "CrossAttentionFusion", "EarlyFusion", "GatedFusion"]
