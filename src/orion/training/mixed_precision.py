"""
Mixed Precision (AMP) Utilities

WHY it exists:
Training large models (Swin, ViT) in Float32 uses too much VRAM and is slow.
Automatic Mixed Precision (AMP) casts operations to Float16 where safe (e.g., Conv2d, Linear)
and keeps them in Float32 where unsafe (e.g., Softmax, BatchNorm).
It cuts memory usage by ~40% and speeds up training by 2x on Tensor Cores.
"""

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from loguru import logger
from typing import Tuple, Any

class AMPManager:
    """
    Wraps PyTorch autocast and GradScaler.
    
    WHY GradScaler? 
    Float16 has a smaller dynamic range. Small gradients can underflow (become 0).
    GradScaler multiplies the loss by a large factor before backward pass,
    pushing gradients into a safe range, then un-scales them before optimizer step.
    """
    def __init__(self, enabled: bool = True, device_type: str = "cuda"):
        self.enabled = enabled
        self.device_type = device_type
        
        # GradScaler is only needed for CUDA float16
        self.scaler = GradScaler() if enabled and device_type == "cuda" else None
        
        dtype = torch.float16 if enabled else torch.float32
        logger.info(f"AMP Manager initialized. Enabled: {enabled}, dtype: {dtype}")
        
    def autocast(self) -> Any:
        """Returns the autocast context manager."""
        return torch.autocast(device_type=self.device_type, enabled=self.enabled, dtype=torch.float16)

    def backward(self, loss: torch.Tensor) -> None:
        """Performs scaled backward pass."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward() # type: ignore
        else:
            loss.backward()
            
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Steps the optimizer and updates scaler."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
