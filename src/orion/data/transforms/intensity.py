
"""
MRI Intensity Transforms
========================

WHY IT EXISTS
-------------

MRI intensity values are not standardized across scanners, protocols, or
acquisitions. Small intensity perturbations can therefore be useful during
training to improve robustness to acquisition differences.

This module contains NumPy-based intensity operations that are useful outside
the Albumentations pipeline.

IMPORTANT
---------

These transforms operate on already-preprocessed image intensities.

The normal project pipeline should generally be:

    DICOM
      ↓
    photometric correction
      ↓
    preprocessing / normalization
      ↓
    intensity augmentation
      ↓
    model

The gamma transform expects values in [0, 1].

This module does NOT perform:

    - DICOM windowing
    - MRI bias-field correction
    - spatial transforms
    - geometric transforms
    - normalization policy selection

Those responsibilities belong elsewhere in the pipeline.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "add_gaussian_noise",
    "gamma_correct",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_volume(
    volume: np.ndarray,
) -> None:
    """
    Validate the common input contract for intensity transforms.
    """

    if not isinstance(
        volume,
        np.ndarray,
    ):
        raise TypeError(
            "volume must be a numpy.ndarray"
        )

    if volume.size == 0:
        raise ValueError(
            "volume must not be empty"
        )

    if not np.issubdtype(
        volume.dtype,
        np.number,
    ):
        raise TypeError(
            f"volume must contain numeric values, got {volume.dtype}"
        )

    if not np.all(
        np.isfinite(volume)
    ):
        raise ValueError(
            "volume contains NaN or infinite values"
        )


# ---------------------------------------------------------------------------
# Gaussian noise
# ---------------------------------------------------------------------------


def add_gaussian_noise(
    volume: np.ndarray,
    std: float,
    rng: np.random.Generator | None = None,
    *,
    clip: bool = False,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to an MRI volume.

    Parameters
    ----------
    volume:
        Numeric MRI image/volume.

    std:
        Standard deviation of the additive Gaussian noise.

        The value uses the same intensity scale as `volume`.

        For a volume normalized to [0, 1], a value such as 0.01 represents
        noise with a standard deviation of roughly 1% of the normalized
        intensity range.

    rng:
        Optional NumPy random generator.

        Supplying an explicit generator allows reproducible augmentation.

    clip:
        If True, clip the result to `value_range`.

        This is useful when the volume is known to be normalized to [0, 1].

        Default is False so this function does not silently alter the
        intensity distribution.

    value_range:
        Inclusive lower/upper range used when `clip=True`.

    Returns
    -------
    np.ndarray
        Float32 volume with the same shape as the input.

    Notes
    -----
    The original volume is never modified in-place.
    """

    _validate_volume(
        volume
    )

    if not isinstance(
        std,
        (float, int),
    ):
        raise TypeError(
            "std must be a real number"
        )

    std = float(
        std
    )

    if not np.isfinite(
        std
    ):
        raise ValueError(
            f"std must be finite, got {std}"
        )

    if std < 0.0:
        raise ValueError(
            f"std must be non-negative, got {std}"
        )

    if not isinstance(
        clip,
        bool,
    ):
        raise TypeError(
            "clip must be a boolean"
        )

    low, high = (
        float(value_range[0]),
        float(value_range[1]),
    )

    if not np.isfinite(
        low
    ) or not np.isfinite(
        high
    ):
        raise ValueError(
            "value_range must contain finite values"
        )

    if low >= high:
        raise ValueError(
            "value_range lower bound must be smaller than upper bound"
        )

    # Always work in float32.
    #
    # This avoids integer overflow/truncation and matches the output of the
    # project's preprocessing normalization.
    base = volume.astype(
        np.float32,
        copy=False,
    )

    if std == 0.0:
        return base.copy()

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=std,
        size=volume.shape,
    ).astype(
        np.float32,
        copy=False,
    )

    result = base + noise

    if clip:
        result = np.clip(
            result,
            low,
            high,
        )

    return result.astype(
        np.float32,
        copy=False,
    )


# ---------------------------------------------------------------------------
# Gamma correction
# ---------------------------------------------------------------------------


def gamma_correct(
    volume: np.ndarray,
    gamma: float,
    *,
    clip: bool = True,
) -> np.ndarray:
    """
    Apply gamma correction to a normalized MRI volume.

    The transform is:

        output = input ** gamma

    Parameters
    ----------
    volume:
        Numeric MRI image/volume expected to be normalized to [0, 1].

    gamma:
        Positive gamma value.

        gamma < 1:
            expands darker intensities.

        gamma > 1:
            suppresses darker intensities.

        gamma == 1:
            identity transform.

    clip:
        If True, clip the input to [0, 1] before applying the transform.

        Default is True.

        This protects against small out-of-range values introduced by
        previous intensity perturbations.

    Returns
    -------
    np.ndarray
        Float32 volume with the same shape as the input.

    Raises
    ------
    ValueError
        If gamma is not finite or is not positive.

    Notes
    -----
    This operation is intended for normalized [0, 1] data. It should not be
    applied directly to arbitrary raw DICOM intensity values.
    """

    _validate_volume(
        volume
    )

    if not isinstance(
        gamma,
        (float, int),
    ):
        raise TypeError(
            "gamma must be a real number"
        )

    gamma = float(
        gamma
    )

    if not np.isfinite(
        gamma
    ):
        raise ValueError(
            f"gamma must be finite, got {gamma}"
        )

    if gamma <= 0.0:
        raise ValueError(
            f"gamma must be positive, got {gamma}"
        )

    if not isinstance(
        clip,
        bool,
    ):
        raise TypeError(
            "clip must be a boolean"
        )

    image = volume.astype(
        np.float32,
        copy=False,
    )

    if clip:
        image = np.clip(
            image,
            0.0,
            1.0,
        )

    return np.power(
        image,
        gamma,
    ).astype(
        np.float32,
        copy=False,
    )

