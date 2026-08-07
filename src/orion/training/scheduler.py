"""
Learning Rate Schedulers

WHY it exists:
Keeping a constant learning rate (LR) is inefficient.
- Too high initially: model diverges. (Solution: Linear warmup)
- Too high later: model bounces around the local minimum. (Solution: Cosine decay)
"""

import math
from torch.optim.lr_scheduler import _LRScheduler # type: ignore
from torch.optim import Optimizer
from ..utils.registry import SCHEDULERS

@SCHEDULERS.register("cosine_with_warmup")
class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine Annealing with a Linear Warmup phase.
    
    WHY: Transformers require warmup. If you hit them with a high LR initially,
    the gradients explode. Cosine decay smoothly reduces the LR to min_lr.
    """
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 1e-7, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self): # type: ignore
        if self.last_epoch < self.warmup_epochs:
            # Linear Warmup
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * alpha 
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine Annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay 
                for base_lr in self.base_lrs
            ]
