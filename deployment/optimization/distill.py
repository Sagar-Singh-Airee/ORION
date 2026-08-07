"""Teacher-to-student loss for compact deployment models."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, temperature: float = 2.0) -> torch.Tensor:
    if not 0 <= alpha <= 1 or temperature <= 0:
        raise ValueError("alpha must be in [0, 1] and temperature positive")
    known = targets >= 0
    supervised = F.binary_cross_entropy_with_logits(student_logits, targets.clamp(0, 1), reduction="none")
    supervised = (supervised * known).sum() / known.sum().clamp_min(1)
    teacher = torch.sigmoid(teacher_logits.detach() / temperature)
    distilled = F.binary_cross_entropy_with_logits(student_logits / temperature, teacher) * temperature**2
    return alpha * supervised + (1 - alpha) * distilled
