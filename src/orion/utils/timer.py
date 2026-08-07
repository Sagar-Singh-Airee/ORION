"""Tiny timing context manager that does not impose a logging backend."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class Timer:
    elapsed: float = field(init=False, default=0.0)
    _start: float = field(init=False, default=0.0)
    def __enter__(self): self._start = perf_counter(); return self
    def __exit__(self, *_): self.elapsed = perf_counter() - self._start
