
"""
MRI Volume Spatial Transforms
=============================

WHY IT EXISTS
-------------

MRI volumes are represented as:

    (D, H, W)

where:

    D = ordered slice axis
    H = image height
    W = image width

Spatial transforms must preserve this contract.

This module therefore contains only transformations that operate within
each 2-D slice and NEVER reorder or modify the slice axis.

IMPORTANT
---------

The first axis is always treated as the slice/depth axis.

For a volume shaped:

    (D, H, W)

the operations are:

    horizontal_flip → W axis
    vertical_flip   → H axis

For arrays with additional trailing dimensions, the same rule is preserved:
the last two dimensions are treated as the spatial image dimensions.

No interpolation or resampling is performed here.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "horizontal_flip",
    "vertical_flip",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_volume(
    volume: np.ndarray,
) -> None:
    """
    Validate the minimum array contract required by spatial transforms.
    """

    if not isinstance(
        volume,
        np.ndarray,
    ):
        raise TypeError(
            "volume must be a numpy.ndarray"
        )

    if volume.ndim < 2:
        raise ValueError(
            "volume must have at least two spatial dimensions"
        )

    if volume.size == 0:
        raise ValueError(
            "volume must not be empty"
        )


# ---------------------------------------------------------------------------
# Horizontal flip
# ---------------------------------------------------------------------------


def horizontal_flip(
    volume: np.ndarray,
) -> np.ndarray:
    """
    Flip every slice horizontally.

    Parameters
    ----------
    volume:
        MRI image/volume.

        For the standard project representation:

            (D, H, W)

        the W axis is reversed.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype.

    IMPORTANT
    ---------
    The slice/depth axis is never modified.
    """

    _validate_volume(
        volume
    )

    # Last axis = image width.
    #
    # np.flip returns a view with negative strides. `.copy()` is intentional:
    # downstream PyTorch/DataLoader code is much safer with a contiguous,
    # independently-owned array.
    return np.flip(
        volume,
        axis=-1,
    ).copy()


# ---------------------------------------------------------------------------
# Vertical flip
# ---------------------------------------------------------------------------


def vertical_flip(
    volume: np.ndarray,
) -> np.ndarray:
    """
    Flip every slice vertically.

    Parameters
    ----------
    volume:
        MRI image/volume.

        For the standard project representation:

            (D, H, W)

        the H axis is reversed.

    Returns
    -------
    np.ndarray
        A new array with the same shape and dtype.

    IMPORTANT
    ---------
    The slice/depth axis is never modified.

    For knee MRI this operation should normally remain disabled in the
    training configuration unless there is a deliberate anatomical
    justification for using it.
    """

    _validate_volume(
        volume
    )

    # Second-to-last axis = image height.
    return np.flip(
        volume,
        axis=-2,
    ).copy()

