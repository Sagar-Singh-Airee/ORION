"""
Timm (PyTorch Image Models) Wrapper

WHY it exists:
Instead of implementing every state-of-the-art model (ConvNeXt, Swin, EfficientNet)
from scratch, we use `timm`. This wrapper provides a unified interface, handles
in-channels (e.g., converting 3-channel pretrained models to accept 1-channel MRI),
and integrates with our registry.
"""

import torch
import torch.nn as nn
from loguru import logger
from ...utils.registry import BACKBONES

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


@BACKBONES.register("timm")
class TimmBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool = True, in_channels: int = 1, drop_path_rate: float = 0.0):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install via `pip install timm`.")
            
        logger.info(f"Loading timm model: {name} (pretrained={pretrained})")
        
        # We use num_classes=0 to get just the feature extractor (removes final classifier)
        # global_pool='' ensures we get the unpooled spatial feature maps (e.g., [B, C, H, W])
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool='',
            drop_path_rate=drop_path_rate
        )
        
        # Get the output feature dimension
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 256, 256)
            dummy_out = self.model(dummy_input)
            
            # Swin transformers output (B, L, C) instead of (B, C, H, W) in some timm versions.
            # Handle shape dynamically.
            if dummy_out.dim() == 4:
                self.out_channels = dummy_out.shape[1]
                self.is_transformer = False
            elif dummy_out.dim() == 3:
                self.out_channels = dummy_out.shape[-1]
                self.is_transformer = True
            else:
                self.out_channels = dummy_out.shape[-1]
        self.model.train(was_training)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            Features. Shape depends on the model:
            CNNs: [B, Channels, H, W]
            ViTs: [B, NumTokens, Channels]
        """
        return self.model(x)


_TIMM_NAME_MAP = {
    "swin_v2_base": "swinv2_base_window12to16_192to256",
    "convnext_large": "convnext_large",
    "efficientnet_b4": "tf_efficientnet_b4",
    "maxvit_base": "maxvit_base_tf_224",
}


def build_timm_backbone(name: str, **kwargs) -> TimmBackbone:
    """Map human-readable experiment names to stable timm model identifiers."""
    return TimmBackbone(name=_TIMM_NAME_MAP.get(name, name), **kwargs)
