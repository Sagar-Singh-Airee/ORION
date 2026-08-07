"""DICOM reading: discover series within a study, order slices correctly, extract the
metadata that mri/sequences.py needs to tell T1 apart from T2/PD/STIR, and mri/anatomy.py
needs to tell sagittal apart from coronal/axial.

The one bug this file exists to prevent: sorting slices by `InstanceNumber` alone.
InstanceNumber is assigned by the scanner/PACS and is not guaranteed to correlate with
spatial position — especially across the 16-institution, multi-vendor set this competition
draws from. Correct ordering is the signed projection of each slice's
ImagePositionPatient onto the series' normal vector (cross product of the row/column
direction cosines from ImageOrientationPatient). InstanceNumber is kept only as a
tie-breaker / fallback when ImagePositionPatient is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError


@dataclass
class SeriesMetadata:
    study_uid: str
    series_uid: str
    series_description: str
    modality: str
    rows: int
    columns: int
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None
    image_orientation_patient: tuple[float, ...] | None  # 6 direction cosines
    repetition_time: float | None   # TR, ms — for sequence ID
    echo_time: float | None         # TE, ms — for sequence ID
    inversion_time: float | None    # TI, ms — distinguishes STIR
    num_slices: int


@dataclass
class DicomSeries:
    metadata: SeriesMetadata
    file_paths: list[Path]          # sorted to match pixel_array order
    pixel_array: np.ndarray         # (num_slices, H, W), rescaled (slope/intercept applied)
    slice_positions: np.ndarray     # (num_slices,) signed projection used for sort order


def _safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_one(path: Path) -> pydicom.Dataset | None:
    try:
        return pydicom.dcmread(str(path), force=True)
    except (InvalidDicomError, OSError):
        return None


def read_series_datasets(file_paths: list[Path]) -> list[tuple[Path, pydicom.Dataset]]:
    """Read only image-bearing files, preserving no assumed input ordering."""
    datasets: list[tuple[Path, pydicom.Dataset]] = []
    for path in file_paths:
        ds = _read_one(path)
        if ds is not None and "PixelData" in ds:
            datasets.append((path, ds))
    return datasets


def discover_series(study_dir: str | Path) -> dict[str, list[Path]]:
    """Walk a study directory and group all readable DICOM files by SeriesInstanceUID."""
    study_dir = Path(study_dir)
    series_map: dict[str, list[Path]] = {}

    for path in study_dir.rglob("*"):
        if not path.is_file():
            continue
        ds = _read_one(path)
        if ds is None or "SeriesInstanceUID" not in ds:
            continue
        series_map.setdefault(ds.SeriesInstanceUID, []).append(path)

    return series_map


def _slice_normal(iop: list[float]) -> np.ndarray:
    row_cosine = np.array(iop[0:3])
    col_cosine = np.array(iop[3:6])
    return np.cross(row_cosine, col_cosine)


def load_series(series_uid: str, file_paths: list[Path]) -> DicomSeries | None:
    """Load, correctly order, and rescale one DICOM series into a (S, H, W) volume."""
    datasets = read_series_datasets(file_paths)

    if not datasets:
        return None

    ref = datasets[0][1]
    iop = getattr(ref, "ImageOrientationPatient", None)

    if iop is not None:
        normal = _slice_normal([float(v) for v in iop])
        keyed = []
        for p, ds in datasets:
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None:
                pos = float(np.dot(normal, [float(v) for v in ipp]))
            else:
                pos = float(getattr(ds, "InstanceNumber", 0))
            keyed.append((pos, p, ds))
        keyed.sort(key=lambda t: t[0])
    else:
        keyed = sorted(
            ((float(getattr(ds, "InstanceNumber", i)), p, ds) for i, (p, ds) in enumerate(datasets)),
            key=lambda t: t[0],
        )

    positions = np.array([k[0] for k in keyed], dtype=np.float32)
    ordered_paths = [k[1] for k in keyed]
    ordered_ds = [k[2] for k in keyed]

    frames = []
    for ds in ordered_ds:
        try:
            arr = ds.pixel_array.astype(np.float32)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Compressed or malformed individual frames should not bring down a
            # preprocessing job. The caller can still use the remaining series.
            return None
        if arr.ndim != 2:
            # Enhanced multi-frame DICOM needs its own geometry handling; treating
            # it as a conventional single frame would silently corrupt ordering.
            return None
        slope = _safe_float(getattr(ds, "RescaleSlope", None), 1.0)
        intercept = _safe_float(getattr(ds, "RescaleIntercept", None), 0.0)
        frames.append(arr * slope + intercept)

    shapes = {f.shape for f in frames}
    if len(shapes) > 1:
        # Mixed in-plane shapes within one series — genuine data-quality issue, drop the
        # series rather than silently resizing (resizing belongs in preprocessor.py, with
        # a logged decision, not hidden here).
        return None

    pixel_array = np.stack(frames, axis=0)

    meta = SeriesMetadata(
        study_uid=str(getattr(ref, "StudyInstanceUID", "")),
        series_uid=series_uid,
        series_description=str(getattr(ref, "SeriesDescription", "")),
        modality=str(getattr(ref, "Modality", "")),
        rows=int(ref.Rows),
        columns=int(ref.Columns),
        pixel_spacing=tuple(float(v) for v in ref.PixelSpacing) if "PixelSpacing" in ref else None,
        slice_thickness=_safe_float(getattr(ref, "SliceThickness", None)),
        image_orientation_patient=tuple(float(v) for v in iop) if iop is not None else None,
        repetition_time=_safe_float(getattr(ref, "RepetitionTime", None)),
        echo_time=_safe_float(getattr(ref, "EchoTime", None)),
        inversion_time=_safe_float(getattr(ref, "InversionTime", None)),
        num_slices=len(ordered_ds),
    )

    return DicomSeries(
        metadata=meta, file_paths=ordered_paths, pixel_array=pixel_array, slice_positions=positions,
    )


def load_study(study_dir: str | Path) -> list[DicomSeries]:
    """All series for one study, loaded and ordered. Series that fail to load
    (unreadable, mixed shapes) are dropped, not raised — a single bad series should not
    take down a whole study; log at the caller if you need visibility into drop rate."""
    series_map = discover_series(study_dir)
    out = []
    for series_uid, paths in series_map.items():
        series = load_series(series_uid, paths)
        if series is not None:
            out.append(series)
    return out
