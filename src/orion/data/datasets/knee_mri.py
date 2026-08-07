"""Kaggle-ready MRI study dataset with real DICOM loading and deterministic sampling."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch

from ...data.cache import NpyDiskCache
from ...data.dicom.preprocessor import preprocess_volume, select_slice_indices
from ...data.dicom.reader import DicomSeries, load_study, read_series_datasets
from ...data.text.label_extractor import FINDINGS
from .base import BaseDataset

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
        self.transform = transform
        self._provided_records = records
        self.label_names = list(_first(_get(config, "labels", {}), "names", default=FINDINGS))
        self.data_cfg = _get(config, "data", config)
        self.root = Path(_first(self.data_cfg, "root", "data_dir", default="."))
        self.preprocess_cfg = _first(self.data_cfg, "preprocessing", default=self.data_cfg)
        self.cache = self._build_cache()
        super().__init__(config, split)

    def _build_cache(self) -> NpyDiskCache | None:
        cache_cfg = _first(self.preprocess_cfg, "cache", default={})
        if not _get(cache_cfg, "enabled", False):
            return None
        backend = str(_get(cache_cfg, "backend", "numpy")).lower()
        if backend not in {"numpy", "npy", "lmdb", "hdf5"}:
            raise ValueError(f"Unsupported cache backend {backend!r}")
        # The portable NPY implementation is deliberate: no LMDB writer locks in
        # Kaggle DataLoader workers. The logical backend name remains accepted.
        directory = _get(cache_cfg, "dir", self.root / ".orion-cache")
        return NpyDiskCache(directory, namespace="dicom-v2")

    def _metadata_path(self) -> Path:
        explicit = _first(
            self.data_cfg,
            f"{self.split}_csv",
            "metadata_csv",
            "labels_csv",
            default=None,
        )
        if explicit is not None:
            path = Path(explicit)
            return path if path.is_absolute() else self.root / path
        candidates = {
            "train": ("train.csv", "train_labels.csv"),
            "val": ("val.csv", "valid.csv", "train.csv"),
            "test": ("test.csv", "sample_submission.csv"),
        }.get(self.split, (f"{self.split}.csv",))
        for name in candidates:
            candidate = self.root / name
            if candidate.exists():
                return candidate
        return self.root / candidates[0]

    def _load_data_records(self) -> list[dict[str, Any]]:
        if self._provided_records is not None:
            return [dict(record) for record in self._provided_records]

        csv_path = self._metadata_path()
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
        split_cfg = _first(self.data_cfg, "splits", default={})
        fold = _get(split_cfg, "fold", None)
        if fold is not None and "fold" in frame.columns and self.split in {"train", "val"}:
            frame = frame[frame["fold"] != int(fold)] if self.split == "train" else frame[frame["fold"] == int(fold)]
        elif self.split == "val" and _get(self.data_cfg, "val_csv", None) is None and "fold" not in frame.columns:
            raise ValueError("Validation requires data.val_csv or a 'fold' column plus data.splits.fold")
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
            labels = self._extract_labels(row)
            records.append(
                {
                    "study_uid": study_uid,
                    "study_path": str(study_path),
                    "label": labels,
                    "report_text": str(row.get(report_column, "") or "") if report_column else "",
                    "metadata": row,
                }
            )
        return records

    def _default_study_path(self, study_uid: str) -> Path:
        image_root = _first(self.data_cfg, f"{self.split}_image_dir", "image_dir", default=None)
        if image_root is not None:
            return Path(image_root) / study_uid
        split_dir = "train_images" if self.split in {"train", "val"} else "test_images"
        return Path(split_dir) / study_uid

    def _extract_labels(self, row: dict[str, Any]) -> list[float]:
        vector_column = _first(self.data_cfg, "label_column", default=None)
        if vector_column and vector_column in row and isinstance(row[vector_column], str):
            parsed = [float(value) for value in row[vector_column].replace("[", "").replace("]", "").split(",")]
            if len(parsed) != len(self.label_names):
                raise ValueError(f"{vector_column} has {len(parsed)} values; expected {len(self.label_names)}")
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
        return f"{record['study_uid']}:{self._target_slices()}:{self._image_size()}"

    def _target_slices(self) -> int:
        selection = _first(self.preprocess_cfg, "slice_selection", default={})
        return int(_first(selection, "num_slices", default=_first(self.data_cfg, "num_slices", default=24)))

    def _image_size(self) -> tuple[int, int]:
        output = _first(self.preprocess_cfg, "output", default={})
        size = _first(output, "image_size", default=_first(self.data_cfg, "image_size", default=(384, 384)))
        return int(size[0]), int(size[1])

    def _select_series(self, series: list[DicomSeries]) -> DicomSeries:
        if not series:
            raise ValueError("Study contains no readable 2-D DICOM series")
        preferred = _first(_first(self.preprocess_cfg, "series_selection", default={}), "preferred_sequences", default=[])
        preferred = [str(item).lower() for item in preferred]
        ranked = sorted(
            series,
            key=lambda item: (
                sum(token in item.metadata.series_description.lower() for token in preferred),
                item.metadata.num_slices,
            ),
            reverse=True,
        )
        return ranked[0]

    def _load_volume(self, record: dict[str, Any]) -> tuple[np.ndarray, int, str]:
        key = self._cache_key(record)
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            return cached, min(len(cached), self._target_slices()), "cached"
        study_path = Path(record["study_path"])
        if not study_path.exists():
            raise FileNotFoundError(f"DICOM study directory does not exist: {study_path}")
        selected = self._select_series(load_study(study_path))
        datasets = [dataset for _, dataset in read_series_datasets(selected.file_paths)]
        volume = preprocess_volume(selected.pixel_array, datasets, self.preprocess_cfg)
        original_count = len(volume)
        selection = _first(self.preprocess_cfg, "slice_selection", default={})
        indices = select_slice_indices(original_count, self._target_slices(), str(_get(selection, "strategy", "uniform")))
        volume = volume[indices]
        if self.cache:
            self.cache.set(key, volume)
        return volume, len(indices), selected.metadata.series_uid

    def _apply_transform(self, volume: np.ndarray) -> np.ndarray:
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
        record = self.data_records[idx]
        volume, real_slices, series_uid = self._load_volume(record)
        volume = self._apply_transform(volume)
        target_slices = self._target_slices()
        if len(volume) < target_slices:
            padded = np.zeros((target_slices, *volume.shape[1:]), dtype=np.float32)
            padded[: len(volume)] = volume
            volume = padded
        elif len(volume) > target_slices:
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
