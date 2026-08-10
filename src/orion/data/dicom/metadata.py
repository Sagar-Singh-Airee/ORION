"""
DICOM Metadata Extractor

WHY it exists:
Often we need to route data differently depending on the scanner manufacturer,
magnetic field strength (1.5T vs 3.0T), or sequence type (T1, T2, PD).
This module extracts that metadata for analysis or conditional processing.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import pydicom

__all__ = ["extract_study_metadata", "identify_sequence_type", "classify_sequence", "SequenceComponents"]


def _as_str(value: Any, default: str = "UNKNOWN") -> str:
    """Normalize to a plain str, treating None/empty/whitespace-only as missing.

    A DICOM tag can be *present* with an empty value (e.g. de-identified data) —
    plain `getattr(ds, tag, default)` only catches a tag that's absent entirely,
    not one that's present-but-blank, so that case is handled here explicitly.
    """
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _as_float(value: Any, default: float = -1.0) -> float:
    """Normalize to a plain float, catching pydicom's numeric wrapper types, None, and ''."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_float_list(value: Any) -> list[float]:
    """Normalize a multi-valued DICOM field (e.g. PixelSpacing) to a plain list[float].

    pydicom represents multi-valued fields as `MultiValue`, which is not JSON-serializable —
    every field returned by this module needs to survive `json.dumps` given how pervasively
    this project writes metadata into manifests/audit trails.
    """
    if value is None:
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def extract_study_metadata(slices: list[pydicom.FileDataset]) -> dict[str, Any]:
    """
    Extracts relevant study and series metadata from the first slice.
    Assuming all slices in a series share these properties — this function does not
    itself verify that assumption (e.g. that every slice shares one SeriesInstanceUID);
    that's the DICOM integrity/validator module's responsibility, not this extractor's.

    Returns a dict with the same fixed set of keys whether or not `slices` is empty,
    so callers never need to special-case an empty result.
    """
    ds = slices[0] if slices else None

    def get(tag: str, default: Any = None) -> Any:
        return getattr(ds, tag, default) if ds is not None else default

    metadata = {
        # Identifiers
        "PatientID": _as_str(get("PatientID")),
        "StudyInstanceUID": _as_str(get("StudyInstanceUID")),
        "SeriesInstanceUID": _as_str(get("SeriesInstanceUID")),
        # Scanner Info
        "Manufacturer": _as_str(get("Manufacturer")),
        "ManufacturerModelName": _as_str(get("ManufacturerModelName")),
        "MagneticFieldStrength": _as_float(get("MagneticFieldStrength")),
        # Sequence Info
        "SeriesDescription": _as_str(get("SeriesDescription")),
        "ProtocolName": _as_str(get("ProtocolName")),
        "ScanningSequence": _as_str(get("ScanningSequence")),
        "SequenceVariant": _as_str(get("SequenceVariant")),
        # Spatial Info
        "SliceThickness": _as_float(get("SliceThickness")),
        "PixelSpacing": _as_float_list(get("PixelSpacing")),
        "ImageOrientationPatient": _as_float_list(get("ImageOrientationPatient")),
        # Volume info
        "NumSlices": len(slices),
    }

    return metadata


class SequenceComponents(NamedTuple):
    """Structured alternative to identify_sequence_type's packed string, for callers
    that want to branch on orientation/weighting/fat-sat directly instead of parsing.
    """

    orientation: str
    weighting: str
    fat_saturated: bool


def classify_sequence(series_description: str | None) -> SequenceComponents:
    """
    Heuristically identifies orientation, contrast weighting, and fat saturation from
    a (messy, vendor-inconsistent) series description.

    WHY: We might want to train models specifically on Sagittal T2, or
    concatenate Sagittal + Coronal + Axial. The series descriptions are messy
    (e.g., "SAG T2 FS", "t2_tse_sag", "Sagittal PD").
    """
    desc = (series_description or "").lower()

    # 1. Identify Orientation
    orientation = "unknown"
    if "sag" in desc:
        orientation = "sagittal"
    elif "cor" in desc:
        orientation = "coronal"
    elif "ax" in desc:
        orientation = "axial"

    # 2. Identify Contrast Weighting. STIR is checked before T1/T2: real-world knee
    # protocols very commonly name STIR series "T2 STIR" / "COR T2 STIR" (STIR has
    # T2-like contrast characteristics), so checking "t2" first would misclassify
    # one of the most clinically important fat-suppressed sequences as plain T2.
    weighting = "unknown"
    if "stir" in desc:
        weighting = "stir"
    elif "t1" in desc:
        weighting = "t1"
    elif "t2" in desc:
        weighting = "t2"
    elif "pd" in desc or "proton" in desc:
        weighting = "pd"

    # 3. Identify Fat Saturation
    fat_saturated = any(token in desc for token in ("fs", "fat", "stir", "dixon", "tirm"))

    return SequenceComponents(orientation=orientation, weighting=weighting, fat_saturated=fat_saturated)


def identify_sequence_type(series_description: str) -> str:
    """
    Heuristically identifies the MRI sequence type and orientation from the description.

    Packed-string form of `classify_sequence`, kept for existing callers/config values
    (e.g. preferred_sequences lists) that already depend on this exact format.
    """
    components = classify_sequence(series_description)
    fat_sat = "yes" if components.fat_saturated else "no"
    return f"{components.orientation}_{components.weighting}_fs={fat_sat}"