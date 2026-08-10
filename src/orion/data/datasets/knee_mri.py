"""Kaggle-ready MRI study dataset with real DICOM loading and deterministic sampling."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ...data.cache import NpyDiskCache
from ...data.dicom.preprocessor import preprocess_volume, select_slice_indices
from ...data.dicom.reader import load_best_series
from ...data.text.label_extractor import FINDINGS
from .base import BaseDataset

__all__ = ["KneeMRIDataset"]

_STUDY_COLUMNS = ("study_uid", "study_id", "StudyInstanceUID", "uid", "id")
_PATH_COLUMNS = ("study_path", "image_path", "path", "directory")
_REPORT_COLUMNS = ("report_text", "report", "text", "radiology_report")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(obj: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = _get(obj, key, None)
        if value is not None:
            return value
    return default


class KneeMRIDataset(BaseDataset):
    """One MRI study as a bag of grayscale slices: ``(S, 1, H, W)``.

    The dataset intentionally does not invent records or synthetic images. A bad
    metadata path or study directory should be caught before a costly Kaggle run.
    It accepts common column spellings so the competition's final CSV can be mapped
    without changing Python code.
    """

    def __init__(
        self,
        config: Any,
        split: str = "train",
        transform: Any | None = None,
        records: list[dict[str, Any]] | None = None,
    ):
        self._provided_records = records
        self.label_names = list(_first(_get(config, "labels", {}), "names", default=FINDINGS))
        self.data_cfg = _get(config, "data", config)
        self.root = Path(_first(self.data_cfg, "root", "data_dir", default="."))
        self.preprocess_cfg = _first(self.data_cfg, "preprocessing", default=self.data_cfg)
        # Recovers a cache hit's real series UID within this process; see _load_volume.
        self._series_uid_by_key: dict[str, str] = {}
        cache = self._build_cache()
        # transform/cache MUST be forwarded here: BaseDataset.__init__ sets
        # self.transform/self._cache itself, so setting them locally and then
        # calling super().__init__(config, split) without forwarding would have
        # them silently overwritten back to None right after construction.
        super().__init__(config, split, transform=transform, cache=cache)

    @property
    def cache(self) -> NpyDiskCache | None:
        """Backward-compatible alias for the base class's injected cache."""
        return self._cache

    def _build_cache(self) -> NpyDiskCache | None:
        cache_cfg = _first(self.preprocess_cfg, "cache", default={})
        if not _get(cache_cfg, "enabled", False):
            return None
        backend = str(_get(cache_cfg, "backend", "numpy")).lower()
        if backend not in {"numpy", "npy"}:
            raise ValueError(
                f"Unsupported cache backend {backend!r}. Only the portable NumPy cache is implemented."
            )
        # NPY files avoid database writer locks in Kaggle DataLoader workers.
        directory = _get(cache_cfg, "dir", self.root / ".orion-cache")
        return NpyDiskCache(directory, namespace="dicom-v2")

    def _metadata_path(self) -> tuple[Path, bool]:
        """Resolve the metadata CSV for `self.split`.

        Returns:
            (path, requires_fold_filter). `requires_fold_filter` is True when the
            resolved file is shared across splits (a generic metadata_csv/labels_csv,
            or falling back to train.csv for the 'val' split) and therefore MUST be
            narrowed by a fold column, as opposed to a file already dedicated to
            this split (e.g. an explicit val_csv, or a found val.csv/valid.csv).
        """
        dedicated = _get(self.data_cfg, f"{self.split}_csv", None)
        if dedicated is not None:
            path = Path(dedicated)
            return (path if path.is_absolute() else self.root / path), False

        shared = _first(self.data_cfg, "metadata_csv", "labels_csv", default=None)
        if shared is not None:
            path = Path(shared)
            return (path if path.is_absolute() else self.root / path), True

        dedicated_candidates = {
            "train": ("train.csv", "train_labels.csv"),
            "val": ("val.csv", "valid.csv"),
            "test": ("test.csv", "sample_submission.csv"),
        }.get(self.split, (f"{self.split}.csv",))
        for name in dedicated_candidates:
            candidate = self.root / name
            if candidate.exists():
                return candidate, False

        if self.split == "val":
            # Nothing dedicated to 'val' exists; falling back to train.csv is only
            # safe if it's then narrowed by a fold column — see _apply_fold_filter.
            return self.root / "train.csv", True

        return self.root / dedicated_candidates[0], False

    def _apply_fold_filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Narrow a metadata table shared across splits down to `self.split` via a fold column.

        Fails closed for 'val': an unfiltered validation split that's silently
        identical to train produces a validation score that looks perfectly fine
        and means nothing, with no error anywhere — the worst kind of leakage bug
        because it's invisible. 'train' is allowed to proceed unfiltered (e.g. a
        final full-data retrain after cross-validation is done), but only with an
        explicit printed note, never silently.
        """
        split_cfg = _first(self.data_cfg, "splits", default={})
        fold = _get(split_cfg, "fold", None)
        has_fold_column = "fold" in frame.columns

        if fold is not None and has_fold_column:
            return frame[frame["fold"] != int(fold)] if self.split == "train" else frame[frame["fold"] == int(fold)]
        if fold is not None and not has_fold_column:
            raise ValueError(f"data.splits.fold={fold!r} is set but no 'fold' column exists in the metadata")
        if has_fold_column:  # fold is None
            if self.split == "val":
                raise ValueError(
                    "Metadata has a 'fold' column but data.splits.fold is not set; refusing to "
                    "silently use the full, unfiltered table as validation data."
                )
            print("Note: 'fold' column present but data.splits.fold is not set; training on the full, unfiltered table.")
            return frame
        if self.split == "val":
            raise ValueError("Validation requires data.val_csv or both a 'fold' column and data.splits.fold")
        return frame

    def _load_data_records(self) -> list[dict[str, Any]]:
        if self._provided_records is not None:
            return [dict(record) for record in self._provided_records]

        csv_path, requires_fold_filter = self._metadata_path()
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No metadata CSV for split={self.split!r}: {csv_path}. "
                "Set data.metadata_csv or data.<split>_csv explicitly."
            )
        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ValueError(f"Metadata CSV is empty: {csv_path}")
        study_column = next((name for name in _STUDY_COLUMNS if name in frame.columns), None)
        if study_column is None:
            raise ValueError(
                f"{csv_path} needs one study-ID column ({', '.join(_STUDY_COLUMNS)}); "
                f"found {list(frame.columns)}"
            )
        if requires_fold_filter and self.split in {"train", "val"}:
            frame = self._apply_fold_filter(frame)
        if frame.empty:
            raise ValueError(f"No records remain for split={self.split!r}")

        records: list[dict[str, Any]] = []
        path_column = next((name for name in _PATH_COLUMNS if name in frame.columns), None)
        report_column = next((name for name in _REPORT_COLUMNS if name in frame.columns), None)
        for row in frame.to_dict("records"):
            study_uid = str(row[study_column])
            raw_path = row.get(path_column) if path_column else None
            study_path = Path(str(raw_path)) if raw_path and not pd.isna(raw_path) else self._default_study_path(study_uid)
            if not study_path.is_absolute():
                study_path = self.root / study_path
            labels = self._extract_labels(row, study_uid)
            records.append(
                {
                    "study_uid": study_uid,
                    "study_path": str(study_path),
                    "label": labels,
                    "report_text": str(row.get(report_column, "") or "") if report_column else "",
                    "metadata": row,
                }
            )

        # A dataset-wide config mismatch (e.g. cfg.labels.names doesn't match the CSV's
        # actual column names) makes every record's label vector fully abstained
        # (-1 for every finding). That's legitimate per-record, but never legitimate
        # for the *entire* train/val split — catch the config bug here instead of
        # letting training silently proceed on a fully-masked loss.
        if self.split in {"train", "val"} and records and all(all(v == -1.0 for v in r["label"]) for r in records):
            raise ValueError(
                f"Every record in {csv_path} has fully-abstained labels ({self.label_names}); "
                "this usually means the label columns don't match cfg.labels.names — check spelling/casing."
            )
        return records

    def _default_study_path(self, study_uid: str) -> Path:
        image_root = _first(self.data_cfg, f"{self.split}_image_dir", "image_dir", default=None)
        if image_root is not None:
            return Path(image_root) / study_uid
        split_dir = "train_images" if self.split in {"train", "val"} else "test_images"
        return Path(split_dir) / study_uid

    def _extract_labels(self, row: dict[str, Any], study_uid: str) -> list[float]:
        vector_column = _first(self.data_cfg, "label_column", default=None)
        if vector_column and vector_column in row and isinstance(row[vector_column], str):
            try:
                parsed = [float(value) for value in row[vector_column].replace("[", "").replace("]", "").split(",")]
            except ValueError as exc:
                raise ValueError(
                    f"Study {study_uid}: could not parse {vector_column}={row[vector_column]!r} as a list of floats"
                ) from exc
            if len(parsed) != len(self.label_names):
                raise ValueError(
                    f"Study {study_uid}: {vector_column} has {len(parsed)} value(s); expected {len(self.label_names)}"
                )
            return parsed
        # ``-1`` deliberately denotes missing/weak-label abstention. Loss functions
        # mask it rather than treating it as a confidently negative finding.
        values: list[float] = []
        for label in self.label_names:
            candidates = (label, label.replace("_", " "), label.upper())
            value = next((row[key] for key in candidates if key in row and not pd.isna(row[key])), -1.0)
            values.append(float(value))
        return values

    def _cache_key(self, record: dict[str, Any]) -> str:
        selection = _first(self.preprocess_cfg, "slice_selection", default={})
        normalisation = _first(self.preprocess_cfg, "normalization", default={})
        return ":".join(
            map(
                str,
                (
                    record["study_path"],
                    self._target_slices(),
                    self._image_size(),
                    _get(selection, "strategy", "uniform"),
                    _get(normalisation, "method", "percentile"),
                    _get(normalisation, "percentile_low", None),
                    _get(normalisation, "percentile_high", None),
                    tuple(_get(normalisation, "target_range", (0.0, 1.0))),
                    _get(self.preprocess_cfg, "apply_voi_lut", True),
                    _get(self.preprocess_cfg, "fix_photometric_interpretation", True),
                    tuple(self._preferred_sequences()),
                ),
            )
        )

    def _target_slices(self) -> int:
        selection = _first(self.preprocess_cfg, "slice_selection", default={})
        return int(_first(selection, "num_slices", default=_first(self.data_cfg, "num_slices", default=24)))

    def _image_size(self) -> tuple[int, int]:
        output = _first(self.preprocess_cfg, "output", default={})
        size = _first(output, "image_size", default=_first(self.data_cfg, "image_size", default=(384, 384)))
        return int(size[0]), int(size[1])

    def _preferred_sequences(self) -> list[str]:
        preferred = _first(_first(self.preprocess_cfg, "series_selection", default={}), "preferred_sequences", default=[])
        return [str(item).lower() for item in preferred]

    def _load_volume(self, record: dict[str, Any]) -> tuple[np.ndarray, int, str]:
        key = self._cache_key(record)
        cached = self._cache_get(key)
        if cached is not None:
            # The disk cache only stores the array; the series UID is recovered from
            # this process's own record of the write for `key`. A hit against an
            # entry written by a *different* process (e.g. a resumed run reusing a
            # prior run's disk cache) won't have it, and falls back to "cached".
            series_uid = self._series_uid_by_key.get(key, "cached")
            return cached, min(len(cached), self._target_slices()), series_uid

        study_path = Path(record["study_path"])
        if not study_path.exists():
            raise FileNotFoundError(f"DICOM study directory does not exist: {study_path}")
        selected = load_best_series(study_path, self._preferred_sequences())
        if selected is None:
            raise ValueError(f"Study contains no readable 2-D DICOM series: {study_path}")
        volume = preprocess_volume(selected.pixel_array, selected.datasets or None, self.preprocess_cfg)
        original_count = len(volume)
        selection = _first(self.preprocess_cfg, "slice_selection", default={})
        indices = select_slice_indices(original_count, self._target_slices(), str(_get(selection, "strategy", "uniform")))
        volume = volume[indices]
        series_uid = selected.metadata.series_uid
        self._series_uid_by_key[key] = series_uid
        self._cache_set(key, volume)
        return volume, len(indices), series_uid

    def _apply_transform(self, volume: np.ndarray) -> np.ndarray:
        if len(volume) == 0:
            raise ValueError("Cannot transform an empty volume (0 slices)")
        if self.transform is None:
            return volume
        # ReplayCompose samples parameters once so an MRI remains a physically
        # coherent volume instead of a stack of independently augmented slices.
        first = self.transform(image=volume[0])
        outputs = [first]
        if isinstance(first, Mapping) and "replay" in first and hasattr(self.transform, "replay"):
            outputs.extend(self.transform.replay(first["replay"], image=slice_) for slice_ in volume[1:])
        else:
            outputs.extend(self.transform(image=slice_) for slice_ in volume[1:])
        transformed = []
        for output in outputs:
            image = output["image"] if isinstance(output, Mapping) else output
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            transformed.append(np.asarray(image, dtype=np.float32))
        return np.stack(transformed, axis=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._get_record(idx)
        volume, real_slices, series_uid = self._load_volume(record)
        volume = self._apply_transform(volume)
        target_slices = self._target_slices()
        if len(volume) < target_slices:
            padded = np.zeros((target_slices, *volume.shape[1:]), dtype=np.float32)
            padded[: len(volume)] = volume
            volume = padded
        elif len(volume) > target_slices:
            # Defensive: _load_volume already caps to target_slices via
            # select_slice_indices, so this should be unreachable in practice.
            volume = volume[:target_slices]
            real_slices = target_slices
        image = torch.from_numpy(volume[:, None, :, :].copy())
        mask = torch.zeros(target_slices, dtype=torch.bool)
        mask[:real_slices] = True
        return {
            "image": image,
            "label": torch.tensor(record.get("label", [-1.0] * len(self.label_names)), dtype=torch.float32),
            "slice_mask": mask,
            "study_uid": str(record["study_uid"]),
            "series_uid": series_uid,
            "report_text": record.get("report_text", ""),
            "metadata": record.get("metadata", {}),
        }