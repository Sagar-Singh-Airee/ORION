"""Model-agnostic slice occlusion attribution."""
from __future__ import annotations

import torch


@torch.inference_mode()
def slice_occlusion_scores(model: torch.nn.Module, image: torch.Tensor, target_class: int, slice_mask: torch.Tensor | None = None) -> torch.Tensor:
    baseline = torch.sigmoid(model(image=image, slice_mask=slice_mask))[:, target_class]
    scores = []
    for index in range(image.shape[1]):
        occluded = image.clone(); occluded[:, index] = 0
        probability = torch.sigmoid(model(image=occluded, slice_mask=slice_mask))[:, target_class]
        scores.append(baseline - probability)
    return torch.stack(scores, dim=1)
