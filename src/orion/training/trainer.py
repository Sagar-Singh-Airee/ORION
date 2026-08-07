"""
Main Training Loop (Trainer)

WHY it exists:
Decouples the boilerplate training logic (epoch loop, gradient accumulation, logging)
from the model definition.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
from loguru import logger

from .mixed_precision import AMPManager
from ..utils.wandb_utils import log_metrics
from ..models.ema import ModelEMA

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        scheduler: Any = None,
        ema: ModelEMA | None = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion.to(device)
        self.device = device
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.ema = ema
        self.config = config
        
        self.epochs = config.training.max_epochs
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.clip_val = config.training.gradient_clip_val
        
        self.amp = AMPManager(
            enabled=config.training.mixed_precision,
            device_type="cuda" if "cuda" in str(device) else "cpu"
        )
        
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            masks = batch.get("slice_mask", None)
            if masks is not None:
                masks = masks.to(self.device, non_blocking=True)
                
            # Forward pass with AMP
            with self.amp.autocast():
                logits = self.model(image=images, slice_mask=masks)
                loss = self.criterion(logits, labels)
                
                # Normalize loss for gradient accumulation
                loss = loss / self.grad_accum_steps
                
            # Backward pass
            self.amp.backward(loss)
            
            # Step and cleanup
            if ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                # Unscale before clipping
                if self.amp.scaler is not None:
                    self.amp.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_val)
                
                self.amp.step(self.optimizer)
                self.optimizer.zero_grad()
                
                if self.ema is not None:
                    self.ema.update(self.model)
                    
                self.global_step += 1
                
            # Logging (un-normalize the loss for display)
            total_loss += (loss.item() * self.grad_accum_steps)
            
            if batch_idx % self.config.logging.wandb.log_frequency == 0:
                log_metrics({"train/batch_loss": loss.item() * self.grad_accum_steps}, step=self.global_step)
                
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
