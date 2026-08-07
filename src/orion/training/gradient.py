from __future__ import annotations

import torch


def clip_gradients(parameters, max_norm: float | None) -> float | None:
    if max_norm is None or max_norm <= 0: return None
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))
