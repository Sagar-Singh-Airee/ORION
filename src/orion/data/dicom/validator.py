"""
DICOM Series Validator
======================

WHY IT EXISTS
-------------

Medical-imaging datasets are frequently inconsistent.

A single MRI series may contain:

    - missing slices
    - duplicated slices
    - corrupted DICOM files
    - inconsistent image dimensions
    - inconsistent pixel spacing
    - inconsistent slice thickness
    - inconsistent orientation
    - mixed SeriesInstanceUID values
    - malformed spatial metadata
    - impossible slice geometry

Allowing such a series into training can cause:

    1. incorrect 3D reconstruction
    2. distorted anatomy
    3. silent label/image mismatches
    4. unstable preprocessing
    5. misleading validation performance
    6. crashes during training

This validator performs deterministic structural and geometric checks before
a DICOM series is accepted by the training pipeline.

IMPORTANT
---------

This module validates structural consistency.

It does NOT attempt to determine whether the anatomy is clinically normal
or abnormal. That is a model/data-labeling responsibility.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pydicom
from loguru import logger


__all__ = [
    "validate_series_consistency",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_SPACING_TOLERANCE = 0.15
_DEFAULT_GEOMETRY_TOLERANCE = 1e-4
_DEFAULT_MAX_REPORTED_ERRORS = 20


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def _finite_float(
    value: object,
) -> float | None:
    """Safely convert a DICOM numeric value to a finite float."""

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(result):
        return None

    return result


def _float_array(
    value: object,
    expected_length: int,
) -> np.ndarray | None:
    """Convert a DICOM multi-value field to a finite NumPy array."""

    try:
        array = np.asarray(
            value,
            dtype=np.float64,
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if array.shape != (
        expected_length,
    ):
        return None

    if not np.all(
        np.isfinite(array)
    ):
        return None

    return array


def _get_pixel_spacing(
    ds: pydicom.Dataset,
) -> tuple[float, float] | None:
    """Return PixelSpacing as (row_spacing, column_spacing)."""

    raw = getattr(
        ds,
        "PixelSpacing",
        None,
    )

    values = _float_array(
        raw,
        2,
    )

    if values is None:
        return None

    if np.any(values <= 0):
        return None

    return (
        float(values[0]),
        float(values[1]),
    )


def _get_orientation(
    ds: pydicom.Dataset,
) -> np.ndarray | None:
    """Return the six ImageOrientationPatient direction cosines."""

    raw = getattr(
        ds,
        "ImageOrientationPatient",
        None,
    )

    orientation = _float_array(
        raw,
        6,
    )

    if orientation is None:
        return None

    row = orientation[:3]
    column = orientation[3:]

    # Direction cosines should be approximately unit vectors.
    row_norm = np.linalg.norm(
        row
    )

    column_norm = np.linalg.norm(
        column
    )

    if (
        row_norm < 1e-8
        or column_norm < 1e-8
    ):
        return None

    if abs(
        row_norm - 1.0
    ) > _DEFAULT_GEOMETRY_TOLERANCE:
        return None

    if abs(
        column_norm - 1.0
    ) > _DEFAULT_GEOMETRY_TOLERANCE:
        return None

    # Row and column directions must be approximately perpendicular.
    if abs(
        float(np.dot(row, column))
    ) > _DEFAULT_GEOMETRY_TOLERANCE:
        return None

    return orientation


def _get_slice_location(
    ds: pydicom.Dataset,
) -> float | None:
    """
    Calculate the signed physical location of a slice.

    The location is the projection of ImagePositionPatient onto the slice
    normal derived from ImageOrientationPatient.

    This is preferable to relying on InstanceNumber because InstanceNumber
    does not guarantee physical ordering.
    """

    position = _float_array(
        getattr(
            ds,
            "ImagePositionPatient",
            None,
        ),
        3,
    )

    orientation = _get_orientation(
        ds
    )

    if (
        position is None
        or orientation is None
    ):
        return None

    row = orientation[:3]
    column = orientation[3:]

    normal = np.cross(
        row,
        column,
    )

    normal_norm = np.linalg.norm(
        normal
    )

    if normal_norm < 1e-8:
        return None

    normal = normal / normal_norm

    return float(
        np.dot(
            position,
            normal,
        )
    )


def _limit_errors(
    errors: list[str],
    maximum: int = _DEFAULT_MAX_REPORTED_ERRORS,
) -> list[str]:
    """Bound validator output so a corrupted series cannot flood logs."""

    if len(errors) <= maximum:
        return errors

    omitted = len(errors) - maximum

    return [
        *errors[:maximum],
        f"... {omitted} additional validation error(s) omitted",
    ]


# ---------------------------------------------------------------------------
# Dimension consistency
# ---------------------------------------------------------------------------


def _check_dimension_consistency(
    slices: list[pydicom.Dataset],
) -> list[str]:
    """Ensure every slice has identical Rows and Columns."""

    if not slices:
        return [
            "Cannot check dimensions of an empty slice list"
        ]

    base_rows = getattr(
        slices[0],
        "Rows",
        None,
    )

    base_cols = getattr(
        slices[0],
        "Columns",
        None,
    )

    if (
        base_rows is None
        or base_cols is None
    ):
        return [
            "First slice is missing Rows/Columns metadata"
        ]

    try:
        base_rows = int(base_rows)
        base_cols = int(base_cols)
    except (
        TypeError,
        ValueError,
    ):
        return [
            "First slice has invalid Rows/Columns metadata"
        ]

    if (
        base_rows <= 0
        or base_cols <= 0
    ):
        return [
            f"First slice has invalid dimensions "
            f"({base_rows}, {base_cols})"
        ]

    missing: list[int] = []
    mismatched: list[
        tuple[int, int, int]
    ] = []

    for index, ds in enumerate(
        slices
    ):

        rows = getattr(
            ds,
            "Rows",
            None,
        )

        cols = getattr(
            ds,
            "Columns",
            None,
        )

        if (
            rows is None
            or cols is None
        ):
            missing.append(
                index
            )
            continue

        try:
            rows = int(rows)
            cols = int(cols)
        except (
            TypeError,
            ValueError,
        ):
            missing.append(
                index
            )
            continue

        if (
            rows != base_rows
            or cols != base_cols
        ):
            mismatched.append(
                (
                    index,
                    rows,
                    cols,
                )
            )

    errors: list[str] = []

    if missing:
        errors.append(
            f"{len(missing)} slice(s) have missing or invalid "
            f"Rows/Columns metadata; indices {missing[:5]}"
        )

    if mismatched:
        sample = [
            (
                index,
                (
                    rows,
                    cols,
                ),
            )
            for index, rows, cols in mismatched[:5]
        ]

        errors.append(
            f"{len(mismatched)} slice(s) have dimensions inconsistent "
            f"with base ({base_rows}, {base_cols}); examples {sample}"
        )

    return errors


# ---------------------------------------------------------------------------
# Pixel spacing
# ---------------------------------------------------------------------------


def _check_pixel_spacing(
    slices: list[pydicom.Dataset],
    tolerance: float = _DEFAULT_SPACING_TOLERANCE,
) -> list[str]:
    """Check that all slices use compatible in-plane pixel spacing."""

    base = _get_pixel_spacing(
        slices[0]
    )

    if base is None:
        return [
            "First slice is missing or has invalid PixelSpacing"
        ]

    missing: list[int] = []
    inconsistent: list[
        tuple[int, tuple[float, float] | None]
    ] = []

    for index, ds in enumerate(
        slices
    ):

        spacing = _get_pixel_spacing(
            ds
        )

        if spacing is None:
            missing.append(
                index
            )
            continue

        row_ratio = abs(
            spacing[0] - base[0]
        ) / max(
            abs(base[0]),
            1e-8,
        )

        col_ratio = abs(
            spacing[1] - base[1]
        ) / max(
            abs(base[1]),
            1e-8,
        )

        if (
            row_ratio > tolerance
            or col_ratio > tolerance
        ):
            inconsistent.append(
                (
                    index,
                    spacing,
                )
            )

    errors: list[str] = []

    if missing:
        errors.append(
            f"{len(missing)} slice(s) have missing or invalid "
            f"PixelSpacing; indices {missing[:5]}"
        )

    if inconsistent:
        errors.append(
            f"{len(inconsistent)} slice(s) have inconsistent "
            f"PixelSpacing relative to base {base}; "
            f"examples {inconsistent[:5]}"
        )

    return errors


# ---------------------------------------------------------------------------
# Slice thickness
# ---------------------------------------------------------------------------


def _check_slice_thickness(
    slices: list[pydicom.Dataset],
    tolerance: float = _DEFAULT_SPACING_TOLERANCE,
) -> list[str]:
    """Check that SliceThickness is positive and reasonably consistent."""

    values: list[
        tuple[int, float | None]
    ] = []

    for index, ds in enumerate(
        slices
    ):
        value = _finite_float(
            getattr(
                ds,
                "SliceThickness",
                None,
            )
        )

        values.append(
            (
                index,
                value,
            )
        )

    known = [
        value
        for _, value in values
        if value is not None
    ]

    if not known:
        # Some valid DICOM exports omit SliceThickness.
        return []

    errors: list[str] = []

    invalid = [
        index
        for index, value in values
        if value is not None
        and value <= 0
    ]

    if invalid:
        errors.append(
            f"{len(invalid)} slice(s) have non-positive "
            f"SliceThickness; indices {invalid[:5]}"
        )

    base = known[0]

    inconsistent = [
        (
            index,
            value,
        )
        for index, value in values
        if value is not None
        and value > 0
        and abs(value - base)
        / max(abs(base), 1e-8)
        > tolerance
    ]

    if inconsistent:
        errors.append(
            f"{len(inconsistent)} slice(s) have inconsistent "
            f"SliceThickness relative to {base:.4f} mm; "
            f"examples {inconsistent[:5]}"
        )

    return errors


# ---------------------------------------------------------------------------
# Orientation consistency
# ---------------------------------------------------------------------------


def _check_orientation_consistency(
    slices: list[pydicom.Dataset],
) -> list[str]:
    """Ensure all slices share the same image orientation."""

    base = _get_orientation(
        slices[0]
    )

    if base is None:
        return [
            "First slice is missing or has invalid "
            "ImageOrientationPatient"
        ]

    inconsistent: list[int] = []
    missing: list[int] = []

    for index, ds in enumerate(
        slices
    ):

        orientation = _get_orientation(
            ds
        )

        if orientation is None:
            missing.append(
                index
            )
            continue

        if not np.allclose(
            orientation,
            base,
            rtol=0.0,
            atol=_DEFAULT_GEOMETRY_TOLERANCE,
        ):
            inconsistent.append(
                index
            )

    errors: list[str] = []

    if missing:
        errors.append(
            f"{len(missing)} slice(s) have missing or invalid "
            f"ImageOrientationPatient; indices {missing[:5]}"
        )

    if inconsistent:
        errors.append(
            f"{len(inconsistent)} slice(s) have inconsistent "
            f"ImageOrientationPatient; indices {inconsistent[:5]}"
        )

    return errors


# ---------------------------------------------------------------------------
# Series identity
# ---------------------------------------------------------------------------


def _check_series_identity(
    slices: list[pydicom.Dataset],
) -> list[str]:
    """Ensure slices belong to one DICOM series."""

    series_uids = {
        str(value).strip()
        for value in (
            getattr(
                ds,
                "SeriesInstanceUID",
                None,
            )
            for ds in slices
        )
        if value is not None
        and str(value).strip()
    }

    study_uids = {
        str(value).strip()
        for value in (
            getattr(
                ds,
                "StudyInstanceUID",
                None,
            )
            for ds in slices
        )
        if value is not None
        and str(value).strip()
    }

    errors: list[str] = []

    if len(series_uids) > 1:
        errors.append(
            "Slices contain multiple SeriesInstanceUID values: "
            f"{len(series_uids)} distinct series detected"
        )

    if len(study_uids) > 1:
        errors.append(
            "Slices contain multiple StudyInstanceUID values: "
            f"{len(study_uids)} distinct studies detected"
        )

    return errors


# ---------------------------------------------------------------------------
# Slice geometry / spacing
# ---------------------------------------------------------------------------


def _check_slice_spacing(
    slices: list[pydicom.Dataset],
) -> list[str]:
    """
    Detect missing or duplicated slices using physical positions.

    The slices are evaluated using their position along the volume normal,
    not InstanceNumber.
    """

    locations = [
        _get_slice_location(ds)
        for ds in slices
    ]

    known = [
        (
            index,
            location,
        )
        for index, location in enumerate(
            locations
        )
        if location is not None
    ]

    if not known:
        # Spatial tags may legitimately be unavailable in some exports.
        return []

    errors: list[str] = []

    missing_indices = [
        index
        for index, location in enumerate(
            locations
        )
        if location is None
    ]

    if missing_indices:
        errors.append(
            f"{len(missing_indices)} of {len(slices)} slice(s) "
            f"have missing or malformed spatial geometry; "
            f"indices {missing_indices[:5]}"
        )

    if len(known) < 2:
        return errors

    positions = np.sort(
        np.asarray(
            [
                location
                for _, location in known
            ],
            dtype=np.float64,
        )
    )

    diffs = np.diff(
        positions
    )

    positive_diffs = diffs[
        diffs > _DEFAULT_GEOMETRY_TOLERANCE
    ]

    if positive_diffs.size == 0:
        errors.append(
            "Slices do not have distinct physical positions"
        )
        return errors

    median_spacing = float(
        np.median(
            positive_diffs
        )
    )

    if median_spacing <= 1e-8:
        return errors

    # Duplicate / near-duplicate positions.
    duplicate_mask = (
        diffs
        <= _DEFAULT_GEOMETRY_TOLERANCE
    )

    duplicate_count = int(
        np.sum(
            duplicate_mask
        )
    )

    if duplicate_count:
        errors.append(
            f"{duplicate_count} duplicate or near-duplicate "
            f"slice position gap(s) detected"
        )

    # Detect unusually large physical gaps.
    #
    # A threshold of 1.5x the median means that approximately one or more
    # expected slice intervals appear to be missing.
    gap_ratios = (
        diffs
        / median_spacing
    )

    gap_mask = (
        gap_ratios > 1.5
    )

    if np.any(
        gap_mask
    ):

        gap_ratios_selected = (
            gap_ratios[
                gap_mask
            ]
        )

        estimated_missing = int(
            np.sum(
                np.maximum(
                    np.round(
                        gap_ratios_selected
                    ).astype(int)
                    - 1,
                    1,
                )
            )
        )

        largest_gap = float(
            np.max(
                gap_ratios_selected
            )
        )

        errors.append(
            f"Large physical slice gap(s) detected: "
            f"{int(np.sum(gap_mask))} gap(s), "
            f"median spacing={median_spacing:.4f}, "
            f"largest gap={largest_gap:.2f}x median, "
            f"approximately {estimated_missing} missing slice(s)"
        )

    return errors


# ---------------------------------------------------------------------------
# Pixel data availability
# ---------------------------------------------------------------------------


def _check_pixel_data(
    slices: list[pydicom.Dataset],
) -> list[str]:
    """Ensure every slice contains usable PixelData."""

    missing: list[int] = []

    for index, ds in enumerate(
        slices
    ):
        if "PixelData" not in ds:
            missing.append(
                index
            )

    if not missing:
        return []

    return [
        f"{len(missing)} slice(s) have no PixelData; "
        f"indices {missing[:5]}"
    ]


# ---------------------------------------------------------------------------
# Basic DICOM sanity
# ---------------------------------------------------------------------------


def _check_basic_dicom_fields(
    slices: list[pydicom.Dataset],
) -> list[str]:
    """Check basic fields required by the image pipeline."""

    errors: list[str] = []

    invalid_modality: list[int] = []

    for index, ds in enumerate(
        slices
    ):

        modality = str(
            getattr(
                ds,
                "Modality",
                "",
            )
        ).strip().upper()

        if modality and modality != "MR":
            invalid_modality.append(
                index
            )

    if invalid_modality:
        errors.append(
            f"{len(invalid_modality)} slice(s) are not MR modality; "
            f"indices {invalid_modality[:5]}"
        )

    return errors


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------


def validate_series_consistency(
    slices: list[pydicom.Dataset],
) -> tuple[bool, list[str]]:
    """
    Validate whether DICOM slices form a structurally consistent MRI volume.

    Returns
    -------

    (is_valid, errors)

    is_valid:
        True only when no structural/geometric validation error is detected.

    errors:
        Human-readable reasons for rejection.

    Validation includes
    -------------------

    - non-empty series
    - PixelData availability
    - Rows/Columns consistency
    - PixelSpacing consistency
    - SliceThickness consistency
    - ImageOrientationPatient consistency
    - StudyInstanceUID consistency
    - SeriesInstanceUID consistency
    - physical slice-position consistency
    - duplicate slice detection
    - large physical-gap detection
    - basic MR modality sanity
    """

    if not slices:
        return (
            False,
            ["Empty slice list"],
        )

    errors: list[str] = []

    checks = (
        _check_basic_dicom_fields,
        _check_pixel_data,
        _check_series_identity,
        _check_dimension_consistency,
        _check_pixel_spacing,
        _check_slice_thickness,
        _check_orientation_consistency,
        _check_slice_spacing,
    )

    for check in checks:

        try:
            errors.extend(
                check(slices)
            )

        except Exception as exc:  # noqa: BLE001
            # A validator must not crash on malformed medical data.
            logger.debug(
                "Validation check {} failed unexpectedly: {}",
                check.__name__,
                exc,
            )

            errors.append(
                f"{check.__name__} could not be completed: "
                f"{exc}"
            )

    errors = _limit_errors(
        errors,
        _DEFAULT_MAX_REPORTED_ERRORS,
    )

    return (
        len(errors) == 0,
        errors,
    )
