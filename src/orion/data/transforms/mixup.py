"""
Batch-Level MixUp for Multi-Label MRI Training
===============================================

WHY IT EXISTS
-------------

MixUp creates additional training examples by interpolating pairs of
images and their targets:

    x_mix = λ * x_a + (1 - λ) * x_b

    y_mix = λ * y_a + (1 - λ) * y_b

For multi-label classification this naturally produces soft targets.

Example:

    y_a = [1, 0, 1]
    y_b = [0, 1, 1]

with λ = 0.7:

    y_mix = [0.7, 0.3, 1.0]

This is appropriate for BCE-style multi-label training.

IMPORTANT
---------

The project's grouped stratification uses:

    1  = positive
    0  = negative
   -1  = unknown / missing

Unknown labels MUST NOT be interpolated.

For that reason this function requires targets to already contain valid
training values in [0, 1].

The caller should mask/remove unknown labels before applying MixUp.

This module performs no loss calculation and no label imputation.
"""

from __future__ import annotations

import torch


__all__ = [
    "mixup_batch",
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> None:
    """Validate the MixUp input contract."""

    if not isinstance(
        images,
        torch.Tensor,
    ):
        raise TypeError(
            "images must be a torch.Tensor"
        )

    if not isinstance(
        targets,
        torch.Tensor,
    ):
        raise TypeError(
            "targets must be a torch.Tensor"
        )

    if images.ndim < 2:
        raise ValueError(
            "images must have a batch dimension and at least one "
            "additional dimension"
        )

    if targets.ndim < 2:
        raise ValueError(
            "targets must have shape (batch, classes)"
        )

    if images.size(0) != targets.size(0):
        raise ValueError(
            "images and targets must have the same batch size: "
            f"{images.size(0)} != {targets.size(0)}"
        )

    if images.size(0) == 0:
        raise ValueError(
            "MixUp cannot operate on an empty batch"
        )

    if not isinstance(
        alpha,
        (float, int),
    ):
        raise TypeError(
            "alpha must be a real number"
        )

    alpha = float(alpha)

    if alpha <= 0.0:
        raise ValueError(
            f"alpha must be positive, got {alpha}"
        )

    if not torch.is_floating_point(
        targets
    ):
        raise TypeError(
            "targets must be floating-point tensors because MixUp "
            "produces soft targets"
        )

    if not torch.isfinite(
        targets
    ).all():
        raise ValueError(
            "targets contain NaN or infinite values"
        )

    # Unknown labels (-1) are valid for the splitting stage, but they are
    # not valid inputs to this interpolation operation.
    #
    # Interpolating:
    #
    #     -1 and 1
    #
    # would produce a meaningless target rather than a soft probability.
    if torch.any(
        targets < 0
    ) or torch.any(
        targets > 1
    ):
        minimum = float(
            targets.min().item()
        )
        maximum = float(
            targets.max().item()
        )

        raise ValueError(
            "MixUp targets must contain values in [0, 1]. "
            "Unknown/missing labels such as -1 must be masked before "
            f"MixUp. Observed range: [{minimum}, {maximum}]"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float,
]:
    """
    Apply MixUp to one training batch.

    Parameters
    ----------
    images:
        Batch of images/volumes.

        The first dimension must be batch size.

    targets:
        Floating-point multi-label targets with shape:

            (batch, num_classes)

        Values must lie in:

            [0, 1]

        Unknown labels represented by -1 must be handled before calling
        this function.

    alpha:
        Beta distribution concentration parameter.

        The common default is:

            alpha = 0.4

        Larger values produce stronger/more consistently mixed examples.

    Returns
    -------
    mixed_images:
        Mixed image batch.

    mixed_targets:
        Mixed soft targets.

    lam:
        The exact λ used for the original batch in:

            λ * original + (1 - λ) * permuted

    Notes
    -----
    The input tensors are never modified in-place.

    If the batch contains fewer than two samples, the original tensors are
    returned unchanged with λ = 1.0.
    """

    _validate_inputs(
        images,
        targets,
        alpha,
    )

    batch_size = images.size(
        0
    )

    if batch_size < 2:
        return (
            images,
            targets,
            1.0,
        )

    # -------------------------------------------------------------------
    # Sample lambda on the same device as the images.
    #
    # Using tensor-valued Beta parameters avoids unnecessary CPU/device
    # transfers when the training batch already lives on CUDA.
    # -------------------------------------------------------------------

    concentration = torch.tensor(
        float(alpha),
        dtype=torch.float32,
        device=images.device,
    )

    beta = torch.distributions.Beta(
        concentration,
        concentration,
    )

    lam = float(
        beta.sample().item()
    )

    # Keep lambda in the mathematically valid interval.
    #
    # Beta(alpha, alpha) already guarantees this, but the explicit check
    # protects against unexpected distribution behavior.
    lam = min(
        1.0,
        max(
            0.0,
            lam,
        ),
    )

    # -------------------------------------------------------------------
    # Random pairing
    # -------------------------------------------------------------------

    permutation = torch.randperm(
        batch_size,
        device=images.device,
    )

    # Targets may live on a different device in unusual training setups.
    # Normally they should match images, but moving the index explicitly
    # keeps the operation safe.
    target_permutation = permutation.to(
        targets.device
    )

    # -------------------------------------------------------------------
    # Interpolation
    # -------------------------------------------------------------------

    mixed_images = (
        images * lam
        + images[permutation]
        * (1.0 - lam)
    )

    mixed_targets = (
        targets * lam
        + targets[target_permutation]
        * (1.0 - lam)
    )

    return (
        mixed_images,
        mixed_targets,
        lam,
    )

