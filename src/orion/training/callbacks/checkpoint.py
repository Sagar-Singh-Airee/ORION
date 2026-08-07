from __future__ import annotations
from pathlib import Path
import torch

class ModelCheckpoint:
    def __init__(self, directory: str | Path, monitor: str = "macro_auc_roc", mode: str = "max"):
        self.directory=Path(directory); self.directory.mkdir(parents=True, exist_ok=True); self.monitor, self.mode, self.best=monitor, mode, None
    def step(self, metrics: dict[str, float], state: dict) -> bool:
        value=metrics[self.monitor]; improved=self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved: self.best=value; torch.save(state, self.directory / "best.pth")
        return improved
