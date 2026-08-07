"""
Exponential Moving Average (EMA) of Model Weights

WHY it exists:
Stochastic Gradient Descent causes model weights to oscillate around the local minimum.
Maintaining an exponentially weighted moving average of the weights over training
creates a "smoothed" model that is significantly more robust and generalizes better,
especially on noisy datasets like medical imaging.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9998, device: Optional[torch.device] = None):
        """
        Args:
            model: The PyTorch model to track.
            decay: The EMA decay rate (alpha). Higher = slower updates (smoother).
        """
        # Create a deep copy of the model for the EMA
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        
        if self.device is not None:
            self.ema_model.to(self.device)
            
        # Disable gradient tracking for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Updates the EMA weights using the current model weights.
        Call this after every optimizer.step().
        
        Formula: ema_weight = decay * ema_weight + (1 - decay) * current_weight
        """
        # Update parameters
        for ema_param, current_param in zip(self.ema_model.parameters(), model.parameters()):
            if current_param.device != ema_param.device:
                current_param = current_param.to(ema_param.device)
            ema_param.data.mul_(self.decay).add_(current_param.data, alpha=1 - self.decay)
            
        # Update buffers (e.g., BatchNorm running stats)
        for ema_buffer, current_buffer in zip(self.ema_model.buffers(), model.buffers()):
            if current_buffer.device != ema_buffer.device:
                current_buffer = current_buffer.to(ema_buffer.device)
            ema_buffer.data.copy_(current_buffer.data)

    def state_dict(self):
        return self.ema_model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
