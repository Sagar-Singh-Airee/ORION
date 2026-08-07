from __future__ import annotations
import torch

class StochasticWeightAveraging:
    def __init__(self, model: torch.nn.Module): self.model=torch.optim.swa_utils.AveragedModel(model)
    def update(self, model: torch.nn.Module) -> None: self.model.update_parameters(model)
