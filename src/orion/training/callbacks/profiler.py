from __future__ import annotations
from time import perf_counter

class StepProfiler:
    def __enter__(self): self.start=perf_counter(); return self
    def __exit__(self, *_): self.seconds=perf_counter()-self.start
