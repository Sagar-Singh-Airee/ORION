"""
Optimizer Factory

WHY it exists:
Different architectures require different optimizers.
- ResNet/CNNs often train best with SGD + Momentum.
- Transformers (Swin, ViT) require AdamW with layer-wise learning rate decay.
This factory handles instantiation based on config.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Iterable
from loguru import logger
from ..utils.registry import OPTIMIZERS

@OPTIMIZERS.register("adamw")
def create_adamw(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Creates AdamW optimizer with layer-wise learning rate decay if specified.
    
    WHY AdamW?
    Adam with decoupled weight decay. Standard Adam applies L2 regularization
    mixed with the gradient updates, which is suboptimal.
    """
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-2)
    betas = config.get("betas", (0.9, 0.999))
    eps = config.get("eps", 1e-8)
    
    # Simple setup without layer decay
    # In production, layer_decay logic separates parameters by depth
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=tuple(betas),
        eps=eps
    )
    
    logger.info(f"Initialized AdamW (lr={lr}, wd={weight_decay})")
    return optimizer


@OPTIMIZERS.register("sgd")
def create_sgd(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = config.get("lr", 1e-2)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 1e-4)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    logger.info(f"Initialized SGD (lr={lr}, momentum={momentum})")
    return optimizer
