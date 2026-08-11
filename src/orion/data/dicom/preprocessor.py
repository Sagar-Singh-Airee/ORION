"""
DICOM Preprocessor
==================

WHY IT EXISTS
-------------

Transforms loaded DICOM volumes into deterministic, numerically stable
arrays suitable for neural-network training.

Responsibilities
----------------

1. Validate the incoming volume.
2. Correct MONOCHROME1 photometric interpretation when requested.
3. Optionally apply DICOM VOI LUT/windowing.
4. Normalize MRI intensity at the VOLUME level.
5. Select a deterministic subset of slices when requested.
6. Resize only the in-plane dimensions.
7. Return float32 output.

IMPORTANT
---------

The reader is responsible for:

    DICOM decoding
    ↓
    slice ordering
    ↓
    RescaleSlope / RescaleIntercept

This module is responsible for:

    photometric correction
    ↓
    optional VOI/windowing
    ↓
    intensity normalization
    ↓
    slice selection
    ↓
    in-plane resizing

MRI does not have CT-like standardized absolute intensity values.
Therefore normalization is performed per volume rather than using a fixed
global HU-style calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pydicom
from loguru import logger
from pydicom.pixel_data_handlers.util import apply_voi_lut


try:
    from scipy.ndimage import zoom
except ImportError:  # pragma: no cover
    zoom = None


__all__ = [
    "fix_photometric_interpretation",
    "apply_windowing",
    "normalize_intensity",
    "select_slice_indices",
    "resize_volume",
    "preprocess_volume",
]


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


def _get(
    config: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a configuration value from either a mapping or an object.

    Supports normal dictionaries as well as OmegaConf-style configuration
    objects.
    """

    if config is None:
        return default

    if isinstance(config, Mapping):
        return config.get(
            key,
            default,
        )

    return getattr(
        config,
        key,
        default,
    )


# ---------------------------------------------------------------------------
# Photometric interpretation
# ---------------------------------------------------------------------------


def fix_photometric_interpretation(
    image: np.ndarray,
    dicom_dataset: pydicom.Dataset,
) -> np.ndarray:
    """
    Convert MONOCHROME1 into the normal increasing-brightness convention.

    MONOCHROME2:
        Higher pixel value → brighter display.

    MONOCHROME1:
        Lower pixel value → brighter display.

    IMPORTANT
    ---------
    The reader has already applied RescaleSlope and RescaleIntercept.

    Therefore inversion cannot safely use:

        (2 ** BitsStored) - 1 - image

    directly on the already-rescaled image.

    Instead, the representable stored-value range is transformed into the
    modality-value domain first, and the inversion is performed around that
    transformed range.

    If metadata is insufficient, a per-volume numerical fallback is used.
    """

    pi = str(
        dicom_dataset.get(
            "PhotometricInterpretation",
            "",
        )
    ).upper()

    if pi != "MONOCHROME1":
        return image

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    bits_stored = dicom_dataset.get(
        "BitsStored",
        None,
    )

    pixel_representation = dicom_dataset.get(
        "PixelRepresentation",
        0,
    )

    slope = dicom_dataset.get(
        "RescaleSlope",
        1.0,
    )

    intercept = dicom_dataset.get(
        "RescaleIntercept",
        0.0,
    )

    try:
        bits = int(bits_stored)
        representation = int(
            pixel_representation
        )
        slope = float(slope)
        intercept = float(intercept)

        if (
            bits <= 0
            or bits > 32
            or not np.isfinite(slope)
            or not np.isfinite(intercept)
            or slope == 0
        ):
            raise ValueError

        if representation == 0:
            stored_min = 0.0
            stored_max = float(
                (2**bits) - 1
            )

        elif representation == 1:
            stored_min = float(
                -(2 ** (bits - 1))
            )
            stored_max = float(
                (2 ** (bits - 1)) - 1
            )

        else:
            raise ValueError

        modality_a = (
            stored_min * slope
            + intercept
        )

        modality_b = (
            stored_max * slope
            + intercept
        )

        modality_min = min(
            modality_a,
            modality_b,
        )

        modality_max = max(
            modality_a,
            modality_b,
        )

        return (
            modality_min
            + modality_max
            - image
        ).astype(
            np.float32,
            copy=False,
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        # Metadata is incomplete or malformed.
        #
        # We deliberately use the current image range only as a last resort.
        # This is preferable to silently applying an incorrect BitsStored
        # assumption to already-rescaled pixels.
        finite = image[
            np.isfinite(image)
        ]

        if finite.size == 0:
            return image.copy()

        image_min = float(
            np.min(finite)
        )
        image_max = float(
            np.max(finite)
        )

        return (
            image_min
            + image_max
            - image
        ).astype(
            np.float32,
            copy=False,
        )


# ---------------------------------------------------------------------------
# VOI / windowing
# ---------------------------------------------------------------------------


def apply_windowing(
    image: np.ndarray,
    dicom_dataset: pydicom.Dataset,
) -> np.ndarray:
    """
    Apply DICOM VOI LUT/windowing when explicitly requested.

    MRI usually benefits more from robust intensity normalization than from
    scanner-provided display windows. Therefore the end-to-end pipeline does
    NOT enable VOI processing by default.

    This function remains available when a dataset/configuration explicitly
    requests it.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # First try the standard DICOM VOI LUT implementation.
    try:
        result = apply_voi_lut(
            image,
            dicom_dataset,
        )

        result = np.asarray(
            result,
            dtype=np.float32,
        )

        if np.all(
            np.isfinite(result)
        ):
            return result

        logger.debug(
            "VOI LUT produced non-finite values; "
            "falling back to raw image."
        )

    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "apply_voi_lut failed ({}); "
            "trying explicit WindowCenter/WindowWidth.",
            exc,
        )

    # Manual windowing fallback.
    center = dicom_dataset.get(
        "WindowCenter",
        None,
    )

    width = dicom_dataset.get(
        "WindowWidth",
        None,
    )

    if center is None or width is None:
        return image

    try:
        if isinstance(
            center,
            pydicom.multival.MultiValue,
        ):
            center = center[0]

        if isinstance(
            width,
            pydicom.multival.MultiValue,
        ):
            width = width[0]

        center = float(center)
        width = float(width)

    except (
        TypeError,
        ValueError,
    ) as exc:

        logger.debug(
            "Could not parse WindowCenter/WindowWidth: {}",
            exc,
        )

        return image

    if (
        not np.isfinite(center)
        or not np.isfinite(width)
        or width <= 0
    ):
        return image

    lower = center - (
        width / 2.0
    )

    upper = center + (
        width / 2.0
    )

    return np.clip(
        image,
        lower,
        upper,
    ).astype(
        np.float32,
        copy=False,
    )


# ---------------------------------------------------------------------------
# Intensity normalization
# ---------------------------------------------------------------------------


def normalize_intensity(
    volume: np.ndarray,
    method: str = "percentile",
    p_low: float = 0.5,
    p_high: float = 99.5,
    target_range: tuple[float, float] = (
        0.0,
        1.0,
    ),
) -> np.ndarray:
    """
    Normalize one complete MRI volume.

    Methods
    -------

    percentile:
        Clip to [p_low, p_high] and scale to target_range.

    zscore:
        Standardize using volume mean and standard deviation.

    global_minmax:
        Scale minimum → target_range[0] and maximum → target_range[1].

    none:
        Return the image without normalization.

    Notes
    -----

    Percentile normalization is the default because MRI signal intensity
    varies between scanners and acquisitions and extreme artifacts should not
    determine the complete dynamic range.
    """

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    if volume.size == 0:
        raise ValueError(
            "Cannot normalize an empty volume"
        )

    if not np.all(
        np.isfinite(volume)
    ):
        raise ValueError(
            "Volume contains NaN or infinite values"
        )

    method = str(
        method
    ).lower().strip()

    low_target = float(
        target_range[0]
    )

    high_target = float(
        target_range[1]
    )

    if not (
        np.isfinite(low_target)
        and np.isfinite(high_target)
    ):
        raise ValueError(
            "target_range must contain finite values"
        )

    if high_target <= low_target:
        raise ValueError(
            "target_range upper bound must be "
            "greater than its lower bound"
        )

    if method == "none":
        return volume.copy()

    if method == "percentile":

        p_low = float(p_low)
        p_high = float(p_high)

        if not (
            0.0 <= p_low < p_high <= 100.0
        ):
            raise ValueError(
                "Percentiles must satisfy "
                "0 <= p_low < p_high <= 100"
            )

        low = float(
            np.percentile(
                volume,
                p_low,
            )
        )

        high = float(
            np.percentile(
                volume,
                p_high,
            )
        )

        if high - low < 1e-6:
            return np.full_like(
                volume,
                low_target,
                dtype=np.float32,
            )

        volume = np.clip(
            volume,
            low,
            high,
        )

        volume = (
            volume - low
        ) / (
            high - low
        )

        volume = (
            volume
            * (
                high_target
                - low_target
            )
            + low_target
        )

        return volume.astype(
            np.float32,
            copy=False,
        )

    if method == "zscore":

        mean = float(
            np.mean(volume)
        )

        std = float(
            np.std(volume)
        )

        if std < 1e-6:
            return np.zeros_like(
                volume,
                dtype=np.float32,
            )

        return (
            (volume - mean) / std
        ).astype(
            np.float32,
            copy=False,
        )

    if method == "global_minmax":

        min_value = float(
            np.min(volume)
        )

        max_value = float(
            np.max(volume)
        )

        if max_value - min_value < 1e-6:
            return np.full_like(
                volume,
                low_target,
                dtype=np.float32,
            )

        volume = (
            (volume - min_value)
            / (max_value - min_value)
        )

        volume = (
            volume
            * (
                high_target
                - low_target
            )
            + low_target
        )

        return volume.astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        f"Unknown normalization method: {method!r}"
    )


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------


def select_slice_indices(
    num_slices: int,
    target_slices: int,
    strategy: str = "uniform",
) -> np.ndarray:
    """
    Select deterministic, unique, ordered slice indices.

    Strategies
    ----------

    all:
        Keep every slice.

    center:
        Keep a contiguous central block.

    uniform:
        Sample approximately uniformly from the entire volume.

    If target_slices >= num_slices, all slices are retained.
    """

    num_slices = int(
        num_slices
    )

    target_slices = int(
        target_slices
    )

    if (
        num_slices <= 0
        or target_slices <= 0
    ):
        return np.empty(
            0,
            dtype=np.int64,
        )

    if (
        strategy == "all"
        or num_slices <= target_slices
    ):
        return np.arange(
            num_slices,
            dtype=np.int64,
        )

    strategy = str(
        strategy
    ).lower().strip()

    if strategy == "center":

        start = (
            num_slices
            - target_slices
        ) // 2

        return np.arange(
            start,
            start + target_slices,
            dtype=np.int64,
        )

    if strategy != "uniform":
        raise ValueError(
            f"Unknown slice-selection strategy: "
            f"{strategy!r}"
        )

    # Because target_slices <= num_slices, the floor-based construction
    # guarantees strictly increasing unique indices while including both
    # endpoints.
    indices = np.floor(
        np.linspace(
            0,
            num_slices - 1,
            target_slices,
        )
    ).astype(
        np.int64
    )

    if (
        len(indices) != target_slices
        or np.any(
            np.diff(indices) <= 0
        )
    ):
        raise RuntimeError(
            "Internal error: uniform slice selection "
            "produced duplicate or unordered indices"
        )

    return indices


# ---------------------------------------------------------------------------
# In-plane resizing
# ---------------------------------------------------------------------------


def resize_volume(
    volume: np.ndarray,
    image_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """
    Resize only H/W dimensions.

    Slice count and slice ordering are preserved.

    Linear interpolation is used for MRI intensity data.
    """

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    if volume.ndim != 3:
        raise ValueError(
            f"Expected (S, H, W) volume, "
            f"got {volume.shape}"
        )

    if len(image_size) != 2:
        raise ValueError(
            "image_size must contain exactly "
            "(height, width)"
        )

    target_h = int(
        image_size[0]
    )

    target_w = int(
        image_size[1]
    )

    if (
        target_h <= 0
        or target_w <= 0
    ):
        raise ValueError(
            "Target image dimensions must be positive"
        )

    if volume.shape[1:] == (
        target_h,
        target_w,
    ):
        return volume.astype(
            np.float32,
            copy=False,
        )

    if zoom is None:
        raise ImportError(
            "scipy is required for resizing "
            "DICOM volumes"
        )

    factors = (
        1.0,
        target_h / volume.shape[1],
        target_w / volume.shape[2],
    )

    resized = zoom(
        volume,
        factors,
        order=1,
        prefilter=False,
    ).astype(
        np.float32,
        copy=False,
    )

    # scipy can occasionally differ by one pixel because of floating-point
    # output-size rounding. Instead of padding the result with artificial
    # zeros, use a final deterministic crop only when necessary.
    if resized.shape[1:] != (
        target_h,
        target_w,
    ):

        if (
            resized.shape[1] < target_h
            or resized.shape[2] < target_w
        ):
            # Extremely unusual rounding case. Re-run with explicit factors
            # rather than injecting black padding into an MRI volume.
            factors = (
                1.0,
                target_h / resized.shape[1],
                target_w / resized.shape[2],
            )

            resized = zoom(
                resized,
                factors,
                order=1,
                prefilter=False,
            ).astype(
                np.float32,
                copy=False,
            )

        resized = resized[
            :,
            :target_h,
            :target_w,
        ]

    if resized.shape[1:] != (
        target_h,
        target_w,
    ):
        raise RuntimeError(
            "Resize failed to produce the requested "
            f"shape {(target_h, target_w)}; "
            f"got {resized.shape[1:]}"
        )

    return resized


# ---------------------------------------------------------------------------
# End-to-end preprocessing
# ---------------------------------------------------------------------------


def preprocess_volume(
    volume: np.ndarray,
    slices: list[pydicom.Dataset] | None,
    config: Any,
) -> np.ndarray:
    """
    Run the complete preprocessing pipeline.

    Pipeline
    --------

        Validate
            ↓
        Photometric correction
            ↓
        Optional VOI LUT
            ↓
        Volume normalization
            ↓
        Optional slice selection
            ↓
        Optional in-plane resize
            ↓
        float32 output
    """

    volume = np.asarray(
        volume,
        dtype=np.float32,
    )

    # -------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------

    if (
        volume.ndim != 3
        or volume.shape[0] == 0
    ):
        raise ValueError(
            "Expected a non-empty "
            f"(S, H, W) volume, got {volume.shape}"
        )

    if volume.shape[1] <= 0 or volume.shape[2] <= 0:
        raise ValueError(
            f"Invalid spatial volume shape: {volume.shape}"
        )

    if not np.all(
        np.isfinite(volume)
    ):
        raise ValueError(
            "Input volume contains NaN or infinite values"
        )

    if slices is not None and len(
        slices
    ) != len(volume):
        raise ValueError(
            "Number of DICOM datasets must match "
            "volume slice count"
        )

    # -------------------------------------------------------------------
    # DICOM-specific corrections
    # -------------------------------------------------------------------

    fix_photometric = bool(
        _get(
            config,
            "fix_photometric_interpretation",
            _get(
                config,
                "fix_photometric",
                True,
            ),
        )
    )

    # MRI default: do not automatically apply scanner display windows.
    apply_voi = bool(
        _get(
            config,
            "apply_voi_lut",
            False,
        )
    )

    if (
        slices is None
        or not (
            fix_photometric
            or apply_voi
        )
    ):
        processed_volume = volume.copy()

    else:

        processed_slices: list[
            np.ndarray
        ] = []

        for image, dataset in zip(
            volume,
            slices,
            strict=True,
        ):

            processed = image.copy()

            if fix_photometric:
                processed = (
                    fix_photometric_interpretation(
                        processed,
                        dataset,
                    )
                )

            if apply_voi:
                processed = apply_windowing(
                    processed,
                    dataset,
                )

            if not np.all(
                np.isfinite(processed)
            ):
                raise ValueError(
                    "DICOM-specific preprocessing produced "
                    "NaN or infinite values"
                )

            processed_slices.append(
                processed
            )

        processed_volume = np.stack(
            processed_slices,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

    # -------------------------------------------------------------------
    # Intensity normalization
    # -------------------------------------------------------------------

    norm_cfg = _get(
        config,
        "normalization",
        config,
    )

    method = str(
        _get(
            norm_cfg,
            "method",
            "percentile",
        )
    ).lower().strip()

    normalize_enabled = bool(
        _get(
            config,
            "normalize",
            True,
        )
    )

    if (
        normalize_enabled
        and method != "none"
    ):

        target_range = _get(
            norm_cfg,
            "target_range",
            (0.0, 1.0),
        )

        if len(target_range) != 2:
            raise ValueError(
                "normalization.target_range "
                "must contain exactly two values"
            )

        processed_volume = (
            normalize_intensity(
                processed_volume,
                method=method,
                p_low=float(
                    _get(
                        norm_cfg,
                        "percentile_low",
                        0.5,
                    )
                ),
                p_high=float(
                    _get(
                        norm_cfg,
                        "percentile_high",
                        99.5,
                    )
                ),
                target_range=(
                    float(
                        target_range[0]
                    ),
                    float(
                        target_range[1]
                    ),
                ),
            )
        )

    elif method == "none":

        processed_volume = (
            processed_volume.astype(
                np.float32,
                copy=False,
            )
        )

    # -------------------------------------------------------------------
    # Optional slice selection
    # -------------------------------------------------------------------

    slice_cfg = _get(
        config,
        "slices",
        {},
    )

    target_slices = _get(
        slice_cfg,
        "target_slices",
        _get(
            config,
            "target_slices",
            None,
        ),
    )

    if target_slices is not None:

        strategy = _get(
            slice_cfg,
            "strategy",
            _get(
                config,
                "slice_strategy",
                "uniform",
            ),
        )

        indices = select_slice_indices(
            num_slices=processed_volume.shape[0],
            target_slices=int(
                target_slices
            ),
            strategy=strategy,
        )

        if len(indices) == 0:
            raise ValueError(
                "Slice selection produced no slices"
            )

        processed_volume = (
            processed_volume[indices]
        )

    # -------------------------------------------------------------------
    # Optional in-plane resizing
    # -------------------------------------------------------------------

    output_cfg = _get(
        config,
        "output",
        {},
    )

    image_size = _get(
        output_cfg,
        "image_size",
        _get(
            config,
            "image_size",
            None,
        ),
    )

    if image_size is not None:

        processed_volume = resize_volume(
            processed_volume,
            image_size,
        )

    # -------------------------------------------------------------------
    # Final safety checks
    # -------------------------------------------------------------------

    processed_volume = (
        processed_volume.astype(
            np.float32,
            copy=False,
        )
    )

    if not np.all(
        np.isfinite(
            processed_volume
        )
    ):
        raise ValueError(
            "Final preprocessed volume contains "
            "NaN or infinite values"
        )

    return processed_volume
