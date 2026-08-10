"""
DICOM Validator

WHY it exists:
Medical datasets are notoriously messy. Series might be missing slices,
contain corrupted files, or have inconsistent voxel spacings.
Validating before training prevents mid-epoch crashes.
"""
from __future__ import annotations

import numpy as np
import pydicom
from loguru import logger

__all__ = ["validate_series_consistency"]


def _check_dimension_consistency(slices: list[pydicom.FileDataset]) -> list[str]:
    """Check every slice's (Rows, Columns) against the first slice's.

    Bounded and deduplicated: a systematically corrupted series would otherwise
    produce one error line per bad slice (up to hundreds), and a first slice that
    is itself missing Rows/Columns would previously cascade into flagging every
    other slice as "inconsistent" against a None baseline.
    """
    base_rows = getattr(slices[0], "Rows", None)
    base_cols = getattr(slices[0], "Columns", None)
    if base_rows is None or base_cols is None:
        return ["First slice is missing Rows/Columns metadata; cannot validate dimension consistency"]

    missing_indices: list[int] = []
    mismatched: list[tuple[int, object, object]] = []
    for i, ds in enumerate(slices):
        rows = getattr(ds, "Rows", None)
        cols = getattr(ds, "Columns", None)
        if rows is None or cols is None:
            missing_indices.append(i)
        elif rows != base_rows or cols != base_cols:
            mismatched.append((i, rows, cols))

    errors = []
    if missing_indices:
        errors.append(
            f"{len(missing_indices)} slice(s) missing Rows/Columns metadata; e.g. indices {missing_indices[:5]}"
        )
    if mismatched:
        sample = [(i, (r, c)) for i, r, c in mismatched[:5]]
        errors.append(
            f"{len(mismatched)} slice(s) have dimensions inconsistent with base ({base_rows}, {base_cols}); "
            f"e.g. {sample}"
        )
    return errors


def _get_slice_location(ds: pydicom.FileDataset) -> float | None:
    """Signed distance of a slice along its volume's normal vector.

    Returns None (rather than raising) for missing, malformed, or non-finite
    spatial tags — a validator that crashes on messy data defeats its own
    stated purpose of catching messy data before it reaches training.
    """
    pos_raw = getattr(ds, "ImagePositionPatient", None)
    ori_raw = getattr(ds, "ImageOrientationPatient", None)
    if pos_raw is None or ori_raw is None:
        return None
    try:
        pos = np.array(pos_raw, dtype=float)
        ori = np.array(ori_raw, dtype=float)
    except (TypeError, ValueError):
        return None
    if pos.shape != (3,) or ori.shape != (6,):
        return None
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(ori))):
        return None
    normal_vec = np.cross(ori[0:3], ori[3:6])
    return float(np.dot(pos, normal_vec))


def _check_slice_spacing(slices: list[pydicom.FileDataset]) -> list[str]:
    """Detect likely missing slices via gaps in physical position along the volume normal.

    Skipped entirely (no error) if no slice in the series has spatial tags at all —
    some export types genuinely don't include them, and that alone isn't a defect.
    But if *some* slices have it and others don't, that's a real partial-corruption
    signal the original (which only checked slice 0) couldn't distinguish from
    "no spatial info anywhere" and would have crashed on instead of reporting.
    """
    errors: list[str] = []
    try:
        locations = [_get_slice_location(ds) for ds in slices]
        known = [loc for loc in locations if loc is not None]
        if not known:
            return errors  # no spatial info anywhere in this series; not an error by itself

        missing_indices = [i for i, loc in enumerate(locations) if loc is None]
        if missing_indices:
            errors.append(
                f"{len(missing_indices)} of {len(slices)} slice(s) are missing or have malformed "
                f"spatial metadata while others have it; e.g. indices {missing_indices[:5]}"
            )
        if len(known) < 2:
            return errors

        diffs = np.diff(np.sort(known))
        median_spacing = float(np.median(np.abs(diffs)))
        if median_spacing <= 1e-6:
            return errors

        ratios = np.abs(diffs) / median_spacing
        gap_mask = ratios > 1.5
        if np.any(gap_mask):
            per_gap_estimate = np.maximum(np.round(ratios[gap_mask]) - 1, 1)
            total_estimate = int(per_gap_estimate.sum())
            errors.append(
                f"Large gap(s) detected between {int(gap_mask.sum())} pair(s) of slices "
                f"(median spacing {median_spacing:.3f}); estimates roughly {total_estimate} missing slice(s) total"
            )
    except Exception as exc:  # noqa: BLE001 - a validator must report, never crash, on malformed data
        logger.debug(f"Slice spacing check raised unexpectedly: {exc}")
        errors.append(f"Could not evaluate slice spacing: {exc}")
    return errors


def validate_series_consistency(slices: list[pydicom.FileDataset]) -> tuple[bool, list[str]]:
    """
    Checks if a list of DICOM slices forms a consistent 3D volume.

    Returns:
        is_valid: bool
        errors: list of error strings
    """
    if not slices:
        return False, ["Empty slice list"]

    errors = _check_dimension_consistency(slices)
    errors.extend(_check_slice_spacing(slices))
    return len(errors) == 0, errors