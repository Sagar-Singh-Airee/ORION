"""Post-training dynamic quantization for CPU deployment experiments."""
from __future__ import annotations

import torch


def dynamic_quantize(model: torch.nn.Module) -> torch.nn.Module:
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)
