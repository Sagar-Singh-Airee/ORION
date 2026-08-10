"""
DICOM Metadata Extractor

WHY it exists:
Often we need to route data differently depending on the scanner manufacturer,
magnetic field strength (1.5T vs 3.0T), or sequence type (T1, T2, PD).
This module extracts that metadata for analysis or conditional processing.
"""
from __future__ import annotations

import re
from typing import Any

import pydicom
from loguru import logger

__all__ = ["extract_study_metadata", "identify_sequence_type"]


def _get_string(ds: pydicom.FileDataset, tag: str, default: str = "UNKNOWN") -> str:
    """Coerce a DICOM tag to a clean string.

    Handles two common real-world artifacts that plain `getattr(ds, tag, default)`
    doesn't: a tag that exists but is empty (frequent after anonymization, which
    often blanks a value rather than removing the tag), and a genuinely
    multi-valued tag (e.g. ScanningSequence=['SE', 'IR']), which pydicom exposes
    as a MultiValue — joined with '\\', DICOM's own convention for multi-valued
    strings, rather than falling through to Python's list repr.
    """
    value = getattr(ds, tag, None)
    if value is None:
        return default
    if isinstance(value, str):
        text = value
    elif hasattr(value, "__iter__"):
        text = "\\".join(str(item) for item in value)
    else:
        text = str(value)
    text = text.strip()
    return text if text else default


def _get_numeric(ds: pydicom.FileDataset, tag: str, default: float = -1.0) -> float:
    """Coerce a DICOM tag to a float, treating an empty-but-present value (same
    anonymization artifact as above) or an unparseable value as missing rather
    than letting it silently reach downstream numeric code as a raw string.
    """
    value = getattr(ds, tag, None)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_float_list(ds: pydicom.FileDataset, tag: str, length: int, sentinel: float = -1.0) -> list[float]:
    """Coerce a multi-valued numeric DICOM tag to a plain list[float] of a fixed length.

    Ensures callers get a consistent shape and type whether or not the tag was
    present: pydicom returns a MultiValue of DSfloat when present, which isn't
    interchangeable with a plain Python list for JSON serialization or vector
    math — and the original code's "missing" default didn't even match the
    "present" case's length (e.g. ImageOrientationPatient: `[]` vs 6 real values).
    """
    default = [sentinel] * length
    value = getattr(ds, tag, None)
    if value is None:
        return default
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return default
    return parsed if len(parsed) == length else default


def extract_study_metadata(slices: list[pydicom.FileDataset]) -> dict[str, Any]:
    """
    Extracts relevant study and series metadata from the first slice.
    Assumes all slices in a series share these properties.
    """
    if not slices:
        raise ValueError("Cannot extract metadata from an empty slice list")

    ds = slices[0]

    return {
        # Identifiers
        "PatientID": _get_string(ds, "PatientID"),
        "StudyInstanceUID": _get_string(ds, "StudyInstanceUID"),
        "SeriesInstanceUID": _get_string(ds, "SeriesInstanceUID"),
        # Scanner Info
        "Manufacturer": _get_string(ds, "Manufacturer"),
        "ManufacturerModelName": _get_string(ds, "ManufacturerModelName"),
        "MagneticFieldStrength": _get_numeric(ds, "MagneticFieldStrength"),
        # Sequence Info
        "SeriesDescription": _get_string(ds, "SeriesDescription"),
        "ProtocolName": _get_string(ds, "ProtocolName"),
        "ScanningSequence": _get_string(ds, "ScanningSequence"),
        "SequenceVariant": _get_string(ds, "SequenceVariant"),
        # Spatial Info
        "SliceThickness": _get_numeric(ds, "SliceThickness"),
        "PixelSpacing": _get_float_list(ds, "PixelSpacing", length=2),
        "ImageOrientationPatient": _get_float_list(ds, "ImageOrientationPatient", length=6),
        # Volume info
        "NumSlices": len(slices),
    }


_TOKEN_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _contains_token(text: str, token: str) -> bool:
    """Match `token` at the start of a word-like chunk (preceded by whitespace,
    underscore, hyphen, or the string start) rather than as a raw substring
    anywhere.

    Prefix matches like 'sag' catching 'sagittal', or 't2' catching 't2_tse_sag',
    still work correctly. What this removes is accidental *mid-word* hits — 'ax'
    inside 'relaxation', 'fs' inside 'offset', 'pd' inside 'update' — which plain
    `token in text` would wrongly count, and which are a real risk given how
    heterogeneous series descriptions are across 16 institutions and 9 languages.

    Patterns are cached since this runs on every series across the whole dataset;
    recompiling per call would undo the point of using regex here at all.

    Note: this cannot distinguish two genuinely different words sharing the same
    prefix (a scanner model name that happens to start with the same letters as
    an orientation/sequence token) — that needs a real vocabulary lookup, not a
    boundary check, and is out of scope for this heuristic.
    """
    pattern = _TOKEN_PATTERN_CACHE.get(token)
    if pattern is None:
        pattern = re.compile(rf"(?:^|[\s_\-]){re.escape(token)}")
        _TOKEN_PATTERN_CACHE[token] = pattern
    return pattern.search(text) is not None


def identify_sequence_type(series_description: str) -> str:
    """
    Heuristically identifies the MRI sequence type and orientation from the description.

    WHY: We might want to train models specifically on Sagittal T2, or
    concatenate Sagittal + Coronal + Axial. The series descriptions are messy
    (e.g., "SAG T2 FS", "t2_tse_sag", "Sagittal PD").
    """
    desc = series_description.lower()

    # 1. Identify Orientation
    orientation = "unknown"
    if _contains_token(desc, "sag"):
        orientation = "sagittal"
    elif _contains_token(desc, "cor"):
        orientation = "coronal"
    elif _contains_token(desc, "ax"):
        orientation = "axial"

    # 2. Identify Contrast Weighting
    weighting = "unknown"
    if _contains_token(desc, "t1"):
        weighting = "t1"
    elif _contains_token(desc, "t2"):
        weighting = "t2"
    elif _contains_token(desc, "pd") or _contains_token(desc, "proton"):
        weighting = "pd"
    elif _contains_token(desc, "stir"):
        weighting = "stir"

    # 3. Identify Fat Saturation
    fat_sat = "no"
    if any(_contains_token(desc, token) for token in ("fs", "fat", "stir", "dixon", "tirm")):
        fat_sat = "yes"

    result = f"{orientation}_{weighting}_fs={fat_sat}"
    if orientation == "unknown" and weighting == "unknown":
        logger.debug(f"Could not classify series description {series_description!r}; heuristic returned {result!r}")
    return result