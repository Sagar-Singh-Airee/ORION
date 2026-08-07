"""Atomic NumPy disk cache suitable for multi-worker PyTorch data loading."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


class NpyDiskCache:
    """Persist arrays atomically using a stable hash of a logical cache key.

    Concurrent workers may race to compute an item, but readers can never observe a
    partial file because the completed temporary file is atomically replaced.
    """

    def __init__(self, directory: str | Path, namespace: str = "v1"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(f"{self.namespace}:{key}".encode("utf-8")).hexdigest()
        return self.directory / f"{digest}.npy"

    def get(self, key: str) -> np.ndarray | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: np.ndarray) -> Path:
        path = self._path(key)
        temp = path.with_suffix(f".{os.getpid()}.tmp.npy")
        try:
            np.save(temp, value, allow_pickle=False)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        return path

    def clear(self) -> int:
        count = 0
        for path in self.directory.glob("*.npy"):
            path.unlink()
            count += 1
        return count
