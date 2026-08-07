"""Generic registry so every component (backbone, loss, fusion, dataset...) is
selectable by a string name from YAML config, without central if/elif chains.

Usage:
    BACKBONES = Registry("backbone")

    @BACKBONES.register("convnext_v2_base")
    class ConvNeXtV2Base(nn.Module): ...

    model_cls = BACKBONES.get(cfg.model.backbone.name)
    model = model_cls(**cfg.model.backbone.kwargs)
"""
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._store: dict[str, type] = {}

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        def _decorator(cls: type[T]) -> type[T]:
            key = name or cls.__name__
            if key in self._store:
                raise KeyError(f"{self.kind} '{key}' already registered "
                                f"(existing: {self._store[key]}, new: {cls})")
            self._store[key] = cls
            return cls
        return _decorator

    def get(self, name: str) -> type:
        if name not in self._store:
            available = ", ".join(sorted(self._store)) or "<empty>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._store[name]

    def build(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def names(self) -> list[str]:
        return sorted(self._store)