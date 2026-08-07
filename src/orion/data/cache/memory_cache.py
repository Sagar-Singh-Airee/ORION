"""Small, thread-safe LRU cache for decoded/preprocessed studies."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class MemoryCache(Generic[K, V]):
    """Bound cache by item count; values are never copied or mutated internally."""

    def __init__(self, max_items: int = 128):
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        self.max_items = max_items
        self._items: OrderedDict[K, V] = OrderedDict()
        self._lock = RLock()

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key not in self._items:
                return default
            self._items.move_to_end(key)
            return self._items[key]

    def set(self, key: K, value: V) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._items
