
"""
MRI-Specific Medical Augmentations
===================================

WHY IT EXISTS
-------------

MRI intensity is affected by acquisition-specific effects that are not
necessarily pathology.

One important example is a smooth intensity bias field:

    observed(x) = true(x) * bias(x)

A model trained only on perfectly uniform intensity can become unnecessarily
sensitive to scanner/coiling/acquisition-related intensity variation.

This module provides conservative MRI-specific perturbations that operate
directly on an already-normalized volume.

IMPORTANT
---------

This module is intentionally separate from:

    - DICOM decoding          -> dicom/
    - preprocessing           -> preprocessor.py
    - generic spatial aug.    -> spatial.py
    - generic intensity aug.  -> intensity.py
    - batch MixUp             -> mixup.py

The functions here should model MRI-specific effects, not generic image
augmentation.

Expected volume layout:

    (S, H, W)

where S is the ordered slice axis.

The slice axis is never modified.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "multiplicative_bias_field",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_volume(
    volume: np.ndarray,
) -> None:
    """Validate the common MRI volume contract."""

    if not isinstance(
        volume,
        np.ndarray,
    ):
        raise TypeError(
            "volume must be a numpy.ndarray"
        )

    if volume.ndim != 3:
        raise ValueError(
            "Expected MRI volume with shape "
            f"(S, H, W), got {volume.shape}"
        )

    if volume.shape[0] == 0:
        raise ValueError(
            "MRI volume must contain at least one slice"
        )

    if volume.shape[1] == 0 or volume.shape[2] == 0:
        raise ValueError(
            "MRI volume must have non-zero spatial dimensions"
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
# Bias field
# ---------------------------------------------------------------------------


def multiplicative_bias_field(
    volume: np.ndarray,
    strength: float = 0.2,
) -> np.ndarray:
    """
    Apply a smooth multiplicative MRI intensity bias field.

    Parameters
    ----------
    volume:
        Normalized MRI volume with shape:

            (S, H, W)

        The function does not modify the input in-place.

    strength:
        Controls the maximum spatial variation.

        Must satisfy:

            0 <= strength <= 1

        A value around 0.1-0.2 represents a conservative perturbation.

        The default is 0.2.

    Returns
    -------
    np.ndarray
        Float32 volume with the same shape as the input.

    Notes
    -----
    The same smooth in-plane field is applied to every slice.

    This is intentional: the augmentation models a slowly varying
    acquisition-related intensity effect rather than independent random
    noise on each slice.

    The generated field is strictly positive and has mean approximately
    one, preventing the augmentation from simply acting as a global
    brightness multiplier.

    This transform is intended for normalized MRI data. It does not perform
    normalization itself.
    """

    _validate_volume(
        volume
    )

    if not isinstance(
        strength,
        (float, int),
    ):
        raise TypeError(
            "strength must be a real number"
        )

    strength = float(
        strength
    )

    if not np.isfinite(
        strength
    ):
        raise ValueError(
            f"strength must be finite, got {strength}"
        )

    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "strength must be in the range [0, 1]"
        )

    image = volume.astype(
        np.float32,
        copy=False,
    )

    if strength == 0.0:
        return image.copy()

    height = volume.shape[1]
    width = volume.shape[2]

    # -------------------------------------------------------------------
    # Normalized spatial coordinates.
    #
    # The coordinate system is independent of the input resolution.
    # -------------------------------------------------------------------

    y = np.linspace(
        -1.0,
        1.0,
        height,
        dtype=np.float32,
    )

    x = np.linspace(
        -1.0,
        1.0,
        width,
        dtype=np.float32,
    )

    yy, xx = np.meshgrid(
        y,
        x,
        indexing="ij",
    )

    # Smooth radial spatial variation.
    #
    # r2 ranges from approximately 0 at the center to 2 at the corners.
    r2 = (
        xx * xx
        + yy * yy
    )

    # Center the spatial pattern so that its mean is approximately zero.
    pattern = r2 - np.mean(
        r2
    )

    # Scale the pattern to a stable range.
    max_abs = float(
        np.max(
            np.abs(pattern)
        )
    )

    if max_abs < 1e-8:
        return image.copy()

    pattern = (
        pattern / max_abs
    ).astype(
        np.float32,
        copy=False,
    )

    # A bounded positive field.
    #
    # The minimum is 1 - strength and the maximum is 1 + strength.
    # Therefore the field remains strictly positive for strength < 1.
    field = (
        1.0
        + strength * pattern
    ).astype(
        np.float32,
        copy=False,
    )

    output = (
        image
        * field[None, :, :]
    )

    return output.astype(
        np.float32,
        copy=False,
    )

