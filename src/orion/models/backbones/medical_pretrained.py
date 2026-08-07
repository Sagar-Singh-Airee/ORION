"""Load a local medical-pretrained checkpoint without network access."""
from __future__ import annotations
from pathlib import Path
import torch
from .timm_wrapper import build_timm_backbone

def load_medical_checkpoint(architecture: str, checkpoint: str | Path, in_channels: int = 1):
    model = build_timm_backbone(architecture, pretrained=False, in_channels=in_channels)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state.get("state_dict", state)), strict=False)
    return model
