"""
DICOM series reader.

WHY IT EXISTS
-------------

Medical imaging datasets frequently contain multiple MRI series per study.
This module discovers those series, reads their DICOM datasets, orders slices
using physical geometry, applies DICOM rescale slope/intercept, and returns
clean 3D volumes.

CRITICAL GEOMETRY RULE
----------------------

Never rely on InstanceNumber alone for slice ordering.

InstanceNumber is a scanner/PACS-assigned value and is not guaranteed to
represent physical slice position.

Preferred ordering:

    ImageOrientationPatient
            ↓
    row/column direction cosines
            ↓
    cross product → slice normal
            ↓
    ImagePositionPatient
            ↓
    signed projection onto normal
            ↓
    physical slice ordering

InstanceNumber is used only as a deterministic tie-breaker or as a fallback
when reliable spatial geometry is unavailable.

The reader intentionally does NOT resize or normalize images. Those operations
belong in preprocessor.py so that physical loading and ML preprocessing remain
separate responsibilities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from loguru import logger


__all__ = [
    "SeriesMetadata",
    "DicomSeries",
    "read_series_datasets",
    "discover_series",
    "load_best_series",
    "load_series",
    "load_study",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesMetadata:
    """Metadata describing a loaded DICOM series."""

    study_uid: str
    series_uid: str
    series_description: str
    modality: str

    rows: int
    columns: int

    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None

    image_orientation_patient: tuple[float, ...] | None

    repetition_time: float | None
    echo_time: float | None
    inversion_time: float | None

    num_slices: int


@dataclass
class DicomSeries:
    """
    Loaded DICOM series.

    Attributes
    ----------
    metadata:
        Series-level DICOM metadata.

    file_paths:
        File paths in exactly the same order as pixel_array.

    pixel_array:
        Float32 volume with shape (S, H, W).

        RescaleSlope and RescaleIntercept have already been applied.

    slice_positions:
        Physical signed position used for ordering.

        If physical geometry was unavailable, this contains deterministic
        fallback ordering positions.

    datasets:
        Ordered DICOM datasets corresponding to file_paths and pixel_array.
    """

    metadata: SeriesMetadata

    file_paths: list[Path]

    pixel_array: np.ndarray

    slice_positions: np.ndarray

    datasets: list[pydicom.Dataset] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_float(
    value: object,
    default: float | None = None,
) -> float | None:
    """Safely convert a DICOM value to float."""

    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(
    value: object,
    default: int,
) -> int:
    """Safely convert a DICOM value to int."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# DICOM reading
# ---------------------------------------------------------------------------


def _read_one(
    path: Path,
    *,
    stop_before_pixels: bool = False,
) -> pydicom.Dataset | None:
    """
    Read one DICOM file.

    A single corrupt/non-DICOM file must not abort discovery of an entire
    medical imaging study.
    """

    try:
        return pydicom.dcmread(
            str(path),
            force=True,
            stop_before_pixels=stop_before_pixels,
        )

    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "Could not read DICOM file {}: {}",
            path,
            exc,
        )
        return None


def read_series_datasets(
    file_paths: list[Path],
) -> list[tuple[Path, pydicom.Dataset]]:
    """
    Read image-bearing DICOM files.

    The input order is deliberately not trusted.
    """

    datasets: list[
        tuple[Path, pydicom.Dataset]
    ] = []

    for path in file_paths:

        ds = _read_one(path)

        if ds is None:
            continue

        if "PixelData" not in ds:
            continue

        datasets.append(
            (path, ds)
        )

    return datasets


# ---------------------------------------------------------------------------
# Series discovery
# ---------------------------------------------------------------------------


def discover_series(
    study_dir: str | Path,
) -> dict[str, list[Path]]:
    """
    Discover image-bearing DICOM files grouped by SeriesInstanceUID.

    Pixel data is skipped during discovery to avoid unnecessary I/O.
    """

    study_dir = Path(study_dir)

    if not study_dir.exists():
        raise FileNotFoundError(
            f"Study directory does not exist: {study_dir}"
        )

    if not study_dir.is_dir():
        raise NotADirectoryError(
            f"Expected study directory, got: {study_dir}"
        )

    series_map: dict[
        str,
        list[Path],
    ] = {}

    for path in study_dir.rglob("*"):

        if not path.is_file():
            continue

        ds = _read_one(
            path,
            stop_before_pixels=True,
        )

        if ds is None:
            continue

        series_uid = getattr(
            ds,
            "SeriesInstanceUID",
            None,
        )

        if not series_uid:
            continue

        # Discovery only returns files that actually contain image data.
        if "PixelData" not in ds:
            # stop_before_pixels=True means PixelData isn't loaded, but the
            # element itself is still available in the dataset when present.
            # Do not reject based on its decoded value.
            pass

        series_map.setdefault(
            str(series_uid),
            [],
        ).append(path)

    return series_map


# ---------------------------------------------------------------------------
# Description matching
# ---------------------------------------------------------------------------


_TOKEN_PATTERN_CACHE: dict[
    str,
    re.Pattern[str],
] = {}


def _contains_token(
    text: str,
    token: str,
) -> bool:
    """
    Match a sequence/orientation token at a word-like boundary.

    Examples
    --------
    "t2_tse_sag" → matches "t2" and "sag"

    "relaxation" → does not accidentally match "ax"

    This avoids dangerous substring matching in heterogeneous
    multi-vendor series descriptions.
    """

    token = token.strip().lower()

    if not token:
        return False

    pattern = _TOKEN_PATTERN_CACHE.get(token)

    if pattern is None:
        pattern = re.compile(
            rf"(?:^|[\s_\-]){re.escape(token)}"
        )

        _TOKEN_PATTERN_CACHE[token] = pattern

    return pattern.search(
        text.lower()
    ) is not None


# ---------------------------------------------------------------------------
# Series selection
# ---------------------------------------------------------------------------


def load_best_series(
    study_dir: str | Path,
    preferred_tokens: list[str] | None = None,
) -> DicomSeries | None:
    """
    Load the most likely useful series.

    Ranking priority:

    1. Number of preferred-token matches.
    2. Number of image files.
    3. Series UID for deterministic ordering.

    Only the selected series is fully decoded.
    """

    preferred = [
        token.lower().strip()
        for token in (preferred_tokens or [])
        if token.strip()
    ]

    candidates: list[
        tuple[
            int,
            int,
            str,
            list[Path],
        ]
    ] = []

    for uid, paths in discover_series(
        study_dir
    ).items():

        header = _read_one(
            paths[0],
            stop_before_pixels=True,
        )

        if header is None:
            continue

        description = str(
            getattr(
                header,
                "SeriesDescription",
                "",
            )
        ).lower()

        score = sum(
            _contains_token(
                description,
                token,
            )
            for token in preferred
        )

        candidates.append(
            (
                score,
                len(paths),
                uid,
                paths,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            item[2],
        )
    )

    for score, _, uid, paths in candidates:

        series = load_series(
            uid,
            paths,
        )

        if series is not None:

            logger.debug(
                "Selected series {} "
                "(token score={}, {} file(s))",
                uid,
                score,
                len(paths),
            )

            return series

    return None


# ---------------------------------------------------------------------------
# Geometry parsing
# ---------------------------------------------------------------------------


def _parse_orientation(
    iop: object,
) -> tuple[float, ...] | None:
    """
    Parse ImageOrientationPatient.

    Returns None when the value is missing, malformed, non-finite, or
    incorrectly shaped.
    """

    if iop is None:
        return None

    try:
        values = tuple(
            float(value)
            for value in iop
        )

    except (TypeError, ValueError):
        return None

    if len(values) != 6:
        return None

    if not all(
        np.isfinite(value)
        for value in values
    ):
        return None

    return values


def _slice_normal(
    iop: tuple[float, ...],
) -> np.ndarray | None:
    """
    Compute the slice normal from ImageOrientationPatient.

    Returns None for degenerate geometry.
    """

    row_cosine = np.asarray(
        iop[0:3],
        dtype=np.float64,
    )

    column_cosine = np.asarray(
        iop[3:6],
        dtype=np.float64,
    )

    normal = np.cross(
        row_cosine,
        column_cosine,
    )

    magnitude = np.linalg.norm(
        normal
    )

    if magnitude < 1e-6:
        return None

    return normal / magnitude


def _position_series(
    datasets: list[
        tuple[Path, pydicom.Dataset]
    ],
    normal: np.ndarray,
) -> list[float] | None:
    """
    Calculate physical slice positions.

    Position is the signed projection of ImagePositionPatient onto the
    normalized slice normal.

    The entire series falls back if any slice lacks valid geometry.
    """

    positions: list[float] = []

    for _, ds in datasets:

        ipp = getattr(
            ds,
            "ImagePositionPatient",
            None,
        )

        if ipp is None:
            return None

        try:
            values = tuple(
                float(value)
                for value in ipp
            )

        except (TypeError, ValueError):
            return None

        if len(values) != 3:
            return None

        if not all(
            np.isfinite(value)
            for value in values
        ):
            return None

        position = float(
            np.dot(
                normal,
                np.asarray(
                    values,
                    dtype=np.float64,
                ),
            )
        )

        if not np.isfinite(position):
            return None

        positions.append(position)

    return positions


# ---------------------------------------------------------------------------
# Fallback ordering
# ---------------------------------------------------------------------------


def _instance_number_positions(
    datasets: list[
        tuple[Path, pydicom.Dataset]
    ],
) -> list[float]:
    """
    Return safe deterministic InstanceNumber values.

    Missing or malformed InstanceNumber values use the original input
    position as a deterministic fallback.
    """

    positions: list[float] = []

    for index, (_, ds) in enumerate(
        datasets
    ):

        instance = _safe_float(
            getattr(
                ds,
                "InstanceNumber",
                None,
            ),
            None,
        )

        if instance is None:
            instance = float(index)

        positions.append(
            instance
        )

    return positions


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------


def _instance_number(
    ds: pydicom.Dataset,
    fallback: int,
) -> int:
    """Return a safe InstanceNumber for deterministic tie-breaking."""

    return _safe_int(
        getattr(
            ds,
            "InstanceNumber",
            None,
        ),
        fallback,
    )


# ---------------------------------------------------------------------------
# Series loading
# ---------------------------------------------------------------------------


def load_series(
    series_uid: str,
    file_paths: list[Path],
) -> DicomSeries | None:
    """
    Load one DICOM series into a correctly ordered (S, H, W) volume.

    Processing performed here:

    - DICOM reading
    - spatial slice ordering
    - deterministic fallback ordering
    - pixel decoding
    - RescaleSlope / RescaleIntercept
    - dimension consistency checks
    - metadata extraction

    Intensity normalization and resizing are intentionally NOT performed here.
    """

    datasets = read_series_datasets(
        file_paths
    )

    if not datasets:
        logger.debug(
            "Series {}: no readable "
            "image-bearing datasets",
            series_uid,
        )
        return None

    # ---------------------------------------------------------------
    # Reference geometry
    # ---------------------------------------------------------------

    ref = datasets[0][1]

    iop = _parse_orientation(
        getattr(
            ref,
            "ImageOrientationPatient",
            None,
        )
    )

    normal = (
        _slice_normal(iop)
        if iop is not None
        else None
    )

    physical_positions = (
        _position_series(
            datasets,
            normal,
        )
        if normal is not None
        else None
    )

    using_physical_order = (
        physical_positions is not None
    )

    if physical_positions is None:
        physical_positions = (
            _instance_number_positions(
                datasets
            )
        )

        logger.debug(
            "Series {}: spatial geometry unavailable; "
            "using InstanceNumber fallback",
            series_uid,
        )

    # ---------------------------------------------------------------
    # Deterministic ordering
    # ---------------------------------------------------------------

    keyed = []

    for index, (
        position,
        pair,
    ) in enumerate(
        zip(
            physical_positions,
            datasets,
            strict=True,
        )
    ):

        path, ds = pair

        instance_number = _instance_number(
            ds,
            index,
        )

        keyed.append(
            (
                float(position),
                instance_number,
                str(path),
                path,
                ds,
            )
        )

    keyed.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    positions_arr = np.asarray(
        [
            item[0]
            for item in keyed
        ],
        dtype=np.float32,
    )

    ordered_paths = [
        item[3]
        for item in keyed
    ]

    ordered_ds = [
        item[4]
        for item in keyed
    ]

    # ---------------------------------------------------------------
    # Validate reference dimensions
    # ---------------------------------------------------------------

    ref_rows = getattr(
        ref,
        "Rows",
        None,
    )

    ref_columns = getattr(
        ref,
        "Columns",
        None,
    )

    if ref_rows is None or ref_columns is None:
        logger.debug(
            "Series {}: missing Rows/Columns metadata",
            series_uid,
        )
        return None

    try:
        ref_rows = int(ref_rows)
        ref_columns = int(ref_columns)

    except (TypeError, ValueError):
        logger.debug(
            "Series {}: invalid Rows/Columns metadata",
            series_uid,
        )
        return None

    if ref_rows <= 0 or ref_columns <= 0:
        logger.debug(
            "Series {}: invalid dimensions {}x{}",
            series_uid,
            ref_rows,
            ref_columns,
        )
        return None

    # ---------------------------------------------------------------
    # Decode pixel data
    # ---------------------------------------------------------------

    frames: list[np.ndarray] = []

    for index, ds in enumerate(
        ordered_ds
    ):

        try:
            arr = np.asarray(
                ds.pixel_array,
                dtype=np.float32,
            )

        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Series {}: could not decode "
                "slice {}: {}",
                series_uid,
                index,
                exc,
            )
            return None

        # Conventional single-frame DICOM only.
        if arr.ndim != 2:
            logger.debug(
                "Series {}: unsupported pixel "
                "array shape {}. Enhanced/multi-frame "
                "DICOM is not handled by this reader.",
                series_uid,
                arr.shape,
            )
            return None

        if arr.shape != (
            ref_rows,
            ref_columns,
        ):
            logger.debug(
                "Series {}: slice {} has shape {}, "
                "expected ({}, {})",
                series_uid,
                index,
                arr.shape,
                ref_rows,
                ref_columns,
            )
            return None

        if not np.all(
            np.isfinite(arr)
        ):
            logger.debug(
                "Series {}: slice {} contains "
                "non-finite pixel values",
                series_uid,
                index,
            )
            return None

        # -----------------------------------------------------------
        # DICOM rescale
        # -----------------------------------------------------------

        slope = _safe_float(
            getattr(
                ds,
                "RescaleSlope",
                None,
            ),
            1.0,
        )

        intercept = _safe_float(
            getattr(
                ds,
                "RescaleIntercept",
                None,
            ),
            0.0,
        )

        if slope is None:
            slope = 1.0

        if intercept is None:
            intercept = 0.0

        arr = (
            arr * slope
            + intercept
        )

        if not np.all(
            np.isfinite(arr)
        ):
            logger.debug(
                "Series {}: rescaling produced "
                "non-finite values in slice {}",
                series_uid,
                index,
            )
            return None

        frames.append(arr)

    if not frames:
        return None

    pixel_array = np.stack(
        frames,
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    # ---------------------------------------------------------------
    # Pixel spacing
    # ---------------------------------------------------------------

    pixel_spacing: tuple[
        float,
        float,
    ] | None = None

    spacing = getattr(
        ref,
        "PixelSpacing",
        None,
    )

    if spacing is not None:

        try:
            values = tuple(
                float(value)
                for value in spacing
            )

            if (
                len(values) == 2
                and all(
                    np.isfinite(value)
                    and value > 0
                    for value in values
                )
            ):
                pixel_spacing = values

        except (TypeError, ValueError):
            pixel_spacing = None

    # ---------------------------------------------------------------
    # Slice thickness
    # ---------------------------------------------------------------

    slice_thickness = _safe_float(
        getattr(
            ref,
            "SliceThickness",
            None,
        )
    )

    if (
        slice_thickness is not None
        and slice_thickness <= 0
    ):
        slice_thickness = None

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    metadata = SeriesMetadata(
        study_uid=str(
            getattr(
                ref,
                "StudyInstanceUID",
                "",
            )
        ),

        series_uid=str(
            series_uid
        ),

        series_description=str(
            getattr(
                ref,
                "SeriesDescription",
                "",
            )
        ),

        modality=str(
            getattr(
                ref,
                "Modality",
                "",
            )
        ),

        rows=ref_rows,
        columns=ref_columns,

        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,

        image_orientation_patient=iop,

        repetition_time=_safe_float(
            getattr(
                ref,
                "RepetitionTime",
                None,
            )
        ),

        echo_time=_safe_float(
            getattr(
                ref,
                "EchoTime",
                None,
            )
        ),

        inversion_time=_safe_float(
            getattr(
                ref,
                "InversionTime",
                None,
            )
        ),

        num_slices=len(
            ordered_ds
        ),
    )

    if not using_physical_order:
        logger.warning(
            "Series {} was loaded using "
            "InstanceNumber fallback because "
            "reliable spatial geometry was unavailable.",
            series_uid,
        )

    return DicomSeries(
        metadata=metadata,
        file_paths=ordered_paths,
        pixel_array=pixel_array,
        slice_positions=positions_arr,
        datasets=ordered_ds,
    )


# ---------------------------------------------------------------------------
# Study loading
# ---------------------------------------------------------------------------


def load_study(
    study_dir: str | Path,
) -> list[DicomSeries]:
    """
    Load all valid DICOM series from a study.

    Invalid/unreadable series are skipped and logged rather than bringing
    down the entire study.
    """

    series_map = discover_series(
        study_dir
    )

    output: list[
        DicomSeries
    ] = []

    for series_uid, paths in sorted(
        series_map.items()
    ):

        series = load_series(
            series_uid,
            paths,
        )

        if series is not None:
            output.append(series)

    return output

