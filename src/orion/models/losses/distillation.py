from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        teacher = torch.sigmoid(teacher_logits / temperature)
        return F.binary_cross_entropy_with_logits(student_logits / temperature, teacher) * temperature**2
