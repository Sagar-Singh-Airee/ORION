"""
Volume-Level MRI Augmentations
==============================

WHY IT EXISTS
-------------

MRI volumes are composed of ordered slices:

    volume.shape == (D, H, W)

where D is the slice axis.

Real MRI acquisitions can contain:

    - motion-corrupted slices
    - low-quality slices
    - partial slice loss
    - inconsistent slice quality

A conservative slice-dropout augmentation can help a model become less
dependent on any single slice.

IMPORTANT
---------

This module operates on the first axis only.

For a standard MRI volume:

    (slices, height, width)

the first dimension is therefore the only dimension modified.

The default behavior is interpolation rather than replacing a slice with
zeros. This avoids introducing artificial black bands that the network could
learn as an augmentation artifact.

Use:

    mode="interpolate"
        Preferred conservative behavior.

    mode="zero"
        Explicitly simulate missing slices as zero-valued slices.

The function never drops every slice.
"""

from __future__ import annotations

from collections.abc import Literal

import numpy as np


__all__ = [
    "random_slice_dropout",
]


DropoutMode = Literal[
    "interpolate",
    "zero",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_volume(
    volume: np.ndarray,
) -> None:
    """Validate the basic volume contract."""

    if not isinstance(
        volume,
        np.ndarray,
    ):
        raise TypeError(
            "volume must be a numpy.ndarray"
        )

    if volume.ndim < 3:
        raise ValueError(
            "volume must have at least 3 dimensions "
            "(slices, height, width)"
        )

    if volume.shape[0] == 0:
        raise ValueError(
            "volume must contain at least one slice"
        )

    if not np.issubdtype(
        volume.dtype,
        np.number,
    ):
        raise TypeError(
            f"volume must contain numeric values, got {volume.dtype}"
        )


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def _interpolate_slice(
    output: np.ndarray,
    target_index: int,
    kept_indices: np.ndarray,
) -> None:
    """
    Replace one dropped slice using its nearest retained neighbors.

    Linear interpolation is used when retained slices exist on both sides.

    At a volume boundary, the nearest retained slice is copied.

    This operates on all remaining dimensions simultaneously, so it works
    for ordinary (D,H,W) MRI volumes as well as volumes with extra channels.
    """

    previous = kept_indices[
        kept_indices < target_index
    ]

    following = kept_indices[
        kept_indices > target_index
    ]

    if (
        previous.size > 0
        and following.size > 0
    ):
        previous_index = int(
            previous[-1]
        )

        following_index = int(
            following[0]
        )

        distance = (
            following_index
            - previous_index
        )

        if distance <= 0:
            output[target_index] = (
                output[previous_index]
            )
            return

        weight = (
            target_index
            - previous_index
        ) / distance

        output[target_index] = (
            output[previous_index]
            * (1.0 - weight)
            + output[following_index]
            * weight
        )

        return

    if previous.size > 0:
        output[target_index] = (
            output[int(previous[-1])]
        )
        return

    if following.size > 0:
        output[target_index] = (
            output[int(following[0])]
        )
        return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def random_slice_dropout(
    volume: np.ndarray,
    probability: float = 0.1,
    rng: np.random.Generator | None = None,
    mode: DropoutMode = "interpolate",
) -> np.ndarray:
    """
    Randomly perturb complete MRI slices along the first axis.

    Parameters
    ----------
    volume:
        MRI volume with shape:

            (D, H, W)

        or any numeric array with the slice dimension first.

    probability:
        Probability that an individual slice is selected.

        Must satisfy:

            0 <= probability < 1

        Probability 0 returns an unchanged copy.

    rng:
        Optional NumPy random generator.

        Supplying one makes the augmentation reproducible.

    mode:
        "interpolate":
            Replace selected slices using neighboring retained slices.

            This is the recommended mode because it avoids artificial
            zero-valued slices.

        "zero":
            Replace selected slices with zeros.

            This explicitly simulates a missing slice and should be used
            deliberately.

    Returns
    -------
    np.ndarray
        Augmented volume with the same shape and dtype as the input.

    Notes
    -----
    At least one slice is always retained.

    For interpolation mode, the function never modifies the original input
    array.
    """

    _validate_volume(
        volume
    )

    if not isinstance(
        probability,
        (float, int),
    ):
        raise TypeError(
            "probability must be a real number"
        )

    probability = float(
        probability
    )

    if not 0.0 <= probability < 1.0:
        raise ValueError(
            "probability must be in [0, 1)"
        )

    if mode not in (
        "interpolate",
        "zero",
    ):
        raise ValueError(
            "mode must be either "
            "'interpolate' or 'zero'"
        )

    # Always return a copy.
    output = volume.copy()

    # Nothing to augment.
    if (
        probability == 0.0
        or len(output) == 1
    ):
        return output

    if rng is None:
        rng = np.random.default_rng()

    # Select slices independently.
    drop_mask = (
        rng.random(
            len(output)
        )
        < probability
    )

    # Never remove every slice.
    if np.all(
        drop_mask
    ):
        keep_index = int(
            rng.integers(
                len(drop_mask)
            )
        )

        drop_mask[
            keep_index
        ] = False

    dropped_indices = np.flatnonzero(
        drop_mask
    )

    if dropped_indices.size == 0:
        return output

    # -------------------------------------------------------------------
    # Explicit zero-fill mode
    # -------------------------------------------------------------------

    if mode == "zero":

        output[
            drop_mask
        ] = 0

        return output

    # -------------------------------------------------------------------
    # Interpolation mode
    # -------------------------------------------------------------------

    kept_indices = np.flatnonzero(
        ~drop_mask
    )

    # Keep the source volume separate from modified output.
    #
    # This is important: if we interpolated directly from `output`, an
    # already-modified slice could become the source for another dropped
    # slice, causing augmentation order to affect the result.
    source = output.copy()

    for index in dropped_indices:

        _interpolate_slice(
            source,
            int(index),
            kept_indices,
        )

    # `_interpolate_slice` writes into its first argument. We want the
    # interpolation source to remain unchanged, so perform the operation
    # into the final output one slice at a time.
    #
    # Recompute from the untouched source to guarantee deterministic,
    # order-independent behavior.
    output = source.copy()

    for index in dropped_indices:

        previous = kept_indices[
            kept_indices < index
        ]

        following = kept_indices[
            kept_indices > index
        ]

        if (
            previous.size > 0
            and following.size > 0
        ):

            previous_index = int(
                previous[-1]
            )

            following_index = int(
                following[0]
            )

            distance = (
                following_index
                - previous_index
            )

            weight = (
                index
                - previous_index
            ) / distance

            interpolated = (
                source[previous_index]
                * (1.0 - weight)
                + source[following_index]
                * weight
            )

            output[index] = (
                interpolated.astype(
                    volume.dtype,
                    copy=False,
                )
            )

        elif previous.size > 0:

            output[index] = source[
                int(previous[-1])
            ]

        elif following.size > 0:

            output[index] = source[
                int(following[0])
            ]

    return output

