"""EMA teacher wrapper for pseudo-label inference."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class TeacherStudent(nn.Module):
    def __init__(self, student: nn.Module, decay: float = 0.999):
        super().__init__()
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.student = student
        self.teacher = copy.deepcopy(student).eval()
        self.decay = decay
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_teacher(self) -> None:
        for teacher, student in zip(self.teacher.parameters(), self.student.parameters(), strict=True):
            teacher.lerp_(student, 1 - self.decay)
        for teacher, student in zip(self.teacher.buffers(), self.student.buffers(), strict=True):
            teacher.copy_(student)
