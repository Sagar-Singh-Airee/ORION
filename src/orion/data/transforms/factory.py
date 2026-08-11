
"""
MRI Training / Validation Augmentation Factories
=================================================

WHY IT EXISTS
-------------

MRI augmentation must improve robustness without creating anatomically
implausible training examples.

This module builds the 2-D augmentation pipeline used on individual MRI
slices.

IMPORTANT
---------

The factory itself only constructs transforms.

For a 3-D MRI volume, the caller must use Albumentations' replay mechanism
to apply the SAME sampled spatial transformation to every slice in the
volume. Otherwise each slice can receive a different rotation/crop/flip and
the resulting volume becomes geometrically inconsistent.

MEDICAL CONSERVATISM
--------------------

The defaults are intentionally conservative:

    - horizontal flip: optional
    - vertical flip: disabled by default
    - small rotation: optional
    - small scale change: optional
    - mild crop: optional
    - mild intensity perturbation: optional
    - no elastic deformation by default
    - no aggressive perspective transformation
    - no arbitrary 3-D deformation

Validation performs resizing only.

This module does not perform tensor conversion or normalization. Those
responsibilities remain in the preprocessing/data pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import numpy as np


try:
    import albumentations as A

    ALBUMENTATIONS_AVAILABLE = True

except ImportError:  # pragma: no cover
    A = None  # type: ignore[assignment]
    ALBUMENTATIONS_AVAILABLE = False


__all__ = [
    "create_train_transforms",
    "create_val_transforms",
]


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _get(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a value from either a mapping or an object.

    This keeps the augmentation factory compatible with ordinary dictionaries
    as well as configuration objects such as OmegaConf namespaces.
    """

    if isinstance(
        obj,
        Mapping,
    ):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def _section(
    config: Any,
) -> Any:
    """
    Resolve the augmentation configuration section.

    Supported structures include:

        config.data.augmentation
        config.augmentation
    """

    data = _get(
        config,
        "data",
        config,
    )

    return _get(
        data,
        "augmentation",
        data,
    )


def _image_size(
    config: Any,
) -> tuple[int, int]:
    """
    Resolve the target 2-D image size.

    Preferred configuration:

        data.preprocessing.output.image_size

    Fallback:

        data.image_size

    Final default:

        (384, 384)
    """

    data = _get(
        config,
        "data",
        config,
    )

    preprocessing = _get(
        data,
        "preprocessing",
        data,
    )

    output = _get(
        preprocessing,
        "output",
        preprocessing,
    )

    size = _get(
        output,
        "image_size",
        _get(
            data,
            "image_size",
            (384, 384),
        ),
    )

    if isinstance(
        size,
        int,
    ):
        return size, size

    if not isinstance(
        size,
        (tuple, list),
    ) or len(size) != 2:
        raise ValueError(
            "image_size must be an integer or a "
            "(height, width) pair"
        )

    height = int(size[0])
    width = int(size[1])

    if height <= 0 or width <= 0:
        raise ValueError(
            f"image_size must be positive, got {(height, width)}"
        )

    return height, width


def _enabled(
    section: Any,
    name: str,
) -> Any | None:
    """
    Return an augmentation configuration only when explicitly enabled.
    """

    value = _get(
        section,
        name,
        {},
    )

    if _get(
        value,
        "enabled",
        False,
    ):
        return value

    return None


def _probability(
    option: Any,
    default: float,
) -> float:
    """
    Validate and return an augmentation probability.
    """

    value = float(
        _get(
            option,
            "p",
            default,
        )
    )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Augmentation probability must be in [0, 1], got {value}"
        )

    return value


def _identity(
    image: np.ndarray | None = None,
    **_: Any,
) -> dict[str, np.ndarray | None]:
    """
    Dependency-free fallback.

    This intentionally performs no augmentation and no resizing because
    silently changing preprocessing behavior when Albumentations is absent
    would be dangerous.
    """

    return {
        "image": image
    }


# ---------------------------------------------------------------------------
# Training transforms
# ---------------------------------------------------------------------------


def create_train_transforms(
    config: Any,
) -> Callable[..., dict[str, Any]]:
    """
    Build the training Albumentations pipeline.

    Returns
    -------
    Callable
        An Albumentations ReplayCompose when Albumentations is installed.

    IMPORTANT FOR 3-D VOLUMES
    -------------------------

    ReplayCompose records the random parameters selected for one image.

    To keep a volume geometrically consistent, the caller must:

        1. transform the first slice
        2. obtain the returned "replay"
        3. replay that transform on every remaining slice

    Do NOT independently call the transform on every slice.

    If Albumentations is unavailable, a no-op transform is returned.
    """

    if not ALBUMENTATIONS_AVAILABLE:
        return _identity

    aug = _section(
        config
    )

    height, width = _image_size(
        config
    )

    spatial = _get(
        aug,
        "spatial",
        {},
    )

    intensity = _get(
        aug,
        "intensity",
        {},
    )

    transforms: list[Any] = []

    # -------------------------------------------------------------------
    # Spatial augmentations
    # -------------------------------------------------------------------

    option = _enabled(
        spatial,
        "hflip",
    )

    if option is not None:
        transforms.append(
            A.HorizontalFlip(
                p=_probability(
                    option,
                    0.5,
                )
            )
        )

    # Vertical flipping is intentionally opt-in.
    #
    # For knee MRI this can reverse the superior/inferior anatomical
    # relationship and is therefore not a conservative default.
    option = _enabled(
        spatial,
        "vflip",
    )

    if option is not None:
        transforms.append(
            A.VerticalFlip(
                p=_probability(
                    option,
                    0.1,
                )
            )
        )

    option = _enabled(
        spatial,
        "rotation",
    )

    if option is not None:

        limit = float(
            _get(
                option,
                "limit",
                10.0,
            )
        )

        if limit < 0:
            raise ValueError(
                "rotation limit must be non-negative"
            )

        transforms.append(
            A.Rotate(
                limit=limit,
                border_mode=0,
                p=_probability(
                    option,
                    0.5,
                ),
            )
        )

    option = _enabled(
        spatial,
        "scale",
    )

    if option is not None:

        scale_limit = float(
            _get(
                option,
                "scale_limit",
                0.1,
            )
        )

        if scale_limit < 0:
            raise ValueError(
                "scale_limit must be non-negative"
            )

        transforms.append(
            A.RandomScale(
                scale_limit=scale_limit,
                p=_probability(
                    option,
                    0.3,
                ),
            )
        )

    # -------------------------------------------------------------------
    # Spatial output size
    # -------------------------------------------------------------------

    crop = _enabled(
        spatial,
        "random_crop",
    )

    if crop is not None:

        scale = tuple(
            float(value)
            for value in _get(
                crop,
                "scale",
                (0.9, 1.0),
            )
        )

        ratio = tuple(
            float(value)
            for value in _get(
                crop,
                "ratio",
                (0.95, 1.05),
            )
        )

        if len(scale) != 2:
            raise ValueError(
                "random_crop.scale must contain two values"
            )

        if len(ratio) != 2:
            raise ValueError(
                "random_crop.ratio must contain two values"
            )

        transforms.append(
            A.RandomResizedCrop(
                height=height,
                width=width,
                scale=scale,
                ratio=ratio,
                p=_probability(
                    crop,
                    1.0,
                ),
            )
        )

    else:

        transforms.append(
            A.Resize(
                height=height,
                width=width,
            )
        )

    # -------------------------------------------------------------------
    # Intensity augmentations
    # -------------------------------------------------------------------

    option = _enabled(
        intensity,
        "brightness_contrast",
    )

    if option is not None:

        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(
                    _get(
                        option,
                        "brightness_limit",
                        0.15,
                    )
                ),
                contrast_limit=float(
                    _get(
                        option,
                        "contrast_limit",
                        0.15,
                    )
                ),
                p=_probability(
                    option,
                    0.5,
                ),
            )
        )

    option = _enabled(
        intensity,
        "gaussian_noise",
    )

    if option is not None:

        variance = _get(
            option,
            "var_limit",
            (5.0, 20.0),
        )

        if not isinstance(
            variance,
            (tuple, list),
        ) or len(variance) != 2:
            raise ValueError(
                "gaussian_noise.var_limit must contain two values"
            )

        transforms.append(
            A.GaussNoise(
                var_limit=tuple(
                    float(value)
                    for value in variance
                ),
                p=_probability(
                    option,
                    0.3,
                ),
            )
        )

    option = _enabled(
        intensity,
        "gaussian_blur",
    )

    if option is not None:

        blur_limit = _get(
            option,
            "blur_limit",
            (3, 5),
        )

        if not isinstance(
            blur_limit,
            (tuple, list),
        ) or len(blur_limit) != 2:
            raise ValueError(
                "gaussian_blur.blur_limit must contain two values"
            )

        transforms.append(
            A.GaussianBlur(
                blur_limit=tuple(
                    int(value)
                    for value in blur_limit
                ),
                p=_probability(
                    option,
                    0.2,
                ),
            )
        )

    option = _enabled(
        intensity,
        "gamma",
    )

    if option is not None:

        gamma_limit = _get(
            option,
            "gamma_limit",
            (80, 120),
        )

        if not isinstance(
            gamma_limit,
            (tuple, list),
        ) or len(gamma_limit) != 2:
            raise ValueError(
                "gamma.gamma_limit must contain two values"
            )

        transforms.append(
            A.RandomGamma(
                gamma_limit=tuple(
                    int(value)
                    for value in gamma_limit
                ),
                p=_probability(
                    option,
                    0.2,
                ),
            )
        )

    # -------------------------------------------------------------------
    # ReplayCompose
    # -------------------------------------------------------------------
    #
    # ReplayCompose is used deliberately because the same sampled spatial
    # transform can be replayed across every slice of a volume.
    #
    # The caller must use the returned "replay" object.
    # -------------------------------------------------------------------

    return A.ReplayCompose(
        transforms
    )


# ---------------------------------------------------------------------------
# Validation transforms
# ---------------------------------------------------------------------------


def create_val_transforms(
    config: Any,
) -> Callable[..., dict[str, Any]]:
    """
    Build the deterministic validation pipeline.

    Validation must not receive stochastic augmentation.

    The only operation performed here is resizing to the configured model
    input size.
    """

    if not ALBUMENTATIONS_AVAILABLE:
        return _identity

    height, width = _image_size(
        config
    )

    return A.Compose(
        [
            A.Resize(
                height=height,
                width=width,
            )
        ]
    )

