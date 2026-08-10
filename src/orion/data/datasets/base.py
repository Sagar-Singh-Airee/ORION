"""
Dataset Base Class

WHY it exists:
Provides common functionality shared by every PyTorch dataset in this project:
parsing the split, applying a shared transform pipeline, an injectable caching hook,
metadata-record loading/validation, and the abstract API contract every dataset
subclass (KneeMRIDataset, MultimodalKneeDataset, ...) must implement.

Design note on caching: this base class defines the *protocol* (get/set), not a
specific backend. Concrete backends (in-memory LRU, LMDB, HDF5 — see
orion.data.cache) are constructed elsewhere and injected via the `cache` argument,
so this file stays decoupled from any particular storage mechanism.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import Dataset

__all__ = ["BaseDataset", "CacheBackend"]


@runtime_checkable
class CacheBackend(Protocol):
    """Minimal duck-typed cache contract. Any object with get/set matching this
    shape can be injected as `cache=...` — an in-memory dict, an LMDB-backed
    disk cache, whatever orion.data.cache provides.
    """

    def get(self, key: Any) -> Optional[Any]: ...

    def set(self, key: Any, value: Any) -> None: ...


class BaseDataset(Dataset, ABC):
    """Abstract base for all ORION datasets.

    Subclasses must implement `_load_data_records` and `__getitem__`. Because
    both are decorated with `@abstractmethod`, Python refuses to instantiate a
    subclass that forgets either one — the bug surfaces at construction time,
    not on the first (possibly much later, possibly on a specific worker
    process) call to `__getitem__`.
    """

    #: Splits this dataset accepts. Subclasses may override if they use
    #: different split names (e.g. a "predict" split), but the default
    #: matches the documented train/val/test contract.
    _VALID_SPLITS: tuple = ("train", "val", "test")

    def __init__(
        self,
        config: DictConfig,
        split: str = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: Optional[CacheBackend] = None,
    ):
        """
        Args:
            config: The data configuration from OmegaConf.
            split: One of `self._VALID_SPLITS` (default: 'train', 'val', 'test').
            transform: Optional callable applied to each sample dict. Subclasses
                should call `self._apply_transform(sample)` as the last step of
                `__getitem__` so transform behavior is consistent across dataset types.
            cache: Optional cache backend (see `CacheBackend`). When provided,
                subclasses may use `self._cache_get` / `self._cache_set` inside
                `__getitem__` to avoid redundant DICOM/text loading. `None` disables
                caching entirely.
        """
        if split not in self._VALID_SPLITS:
            raise ValueError(f"split must be one of {self._VALID_SPLITS}, got {split!r}")

        self.config = config
        self.split = split
        self.transform = transform
        self._cache = cache

        # Subclasses load their own metadata (e.g. a CSV of studies/patients);
        # validated here so a shape bug in that implementation fails immediately
        # and identically for every subclass, rather than surfacing later as a
        # confusing IndexError or KeyError deep inside training.
        self.data_records = self._validate_records(self._load_data_records())

        logger.info(
            f"Initialized {self.__class__.__name__} (split={split}): {len(self)} record(s), "
            f"transform={'enabled' if transform is not None else 'disabled'}, "
            f"cache={'enabled' if cache is not None else 'disabled'}"
        )

    # ------------------------------------------------------------------ #
    # Abstract contract — every subclass must implement these.
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _load_data_records(self) -> List[Dict[str, Any]]:
        """Load the list of studies/patients for `self.split`.

        Returns:
            A non-empty list of dicts, one per example. `self.split` and
            `self.config` are already set when this is called, so it's safe
            to use them to select/filter records for the requested split.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a single example.

        Returns:
            A dict containing at least 'image', 'label', and 'metadata' keys
            (exact contents may vary by subclass, e.g. MultimodalKneeDataset
            also includes tokenized report text). Implementations should use
            `self._get_record(idx)` for bounds-checked metadata access, and
            call `self._apply_transform(sample)` as the final step.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared helpers for subclasses.
    # ------------------------------------------------------------------ #

    def _validate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fail fast on a malformed `_load_data_records` implementation.

        Catches the two bugs that otherwise surface much later and much less
        clearly: an empty split (silently trains/evaluates on nothing) and a
        wrong container/element type (breaks every downstream indexing call).
        """
        if not isinstance(records, list):
            raise TypeError(
                f"{self.__class__.__name__}._load_data_records() must return a list, "
                f"got {type(records).__name__}"
            )
        if not records:
            raise ValueError(
                f"{self.__class__.__name__}._load_data_records() returned 0 records for "
                f"split={self.split!r}; check the configured data paths/filters for this split"
            )
        if not isinstance(records[0], dict):
            raise TypeError(
                f"{self.__class__.__name__}._load_data_records() must return a list of dicts, "
                f"got element of type {type(records[0]).__name__}"
            )
        return records

    def _get_record(self, idx: int) -> Dict[str, Any]:
        """Bounds-checked access to a raw metadata record, for use inside `__getitem__`."""
        try:
            return self.data_records[idx]
        except IndexError as exc:
            raise IndexError(
                f"Index {idx} out of range for {self.__class__.__name__} with {len(self.data_records)} record(s)"
            ) from exc

    def _cache_get(self, key: Any) -> Optional[Any]:
        """Fetch a previously cached item, or None on a miss / when caching is disabled."""
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _cache_set(self, key: Any, value: Any) -> None:
        """Store an item in the cache backend, if one was provided."""
        if self._cache is not None:
            self._cache.set(key, value)

    def _apply_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the configured transform pipeline, if any."""
        return self.transform(sample) if self.transform is not None else sample

    # ------------------------------------------------------------------ #
    # Dunder methods.
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.data_records)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(split={self.split!r}, records={len(self)}, "
            f"transform={'set' if self.transform is not None else None}, "
            f"cache={'set' if self._cache is not None else None})"
        )