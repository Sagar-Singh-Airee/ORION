from __future__ import annotations

class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience, self.mode, self.best, self.wait = patience, mode, None, 0
    def step(self, value: float) -> bool:
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        self.best, self.wait = (value, 0) if improved else (self.best, self.wait + 1)
        return self.wait >= self.patience
