
"""
MRI Data Augmentation Package
=============================

Public augmentation interfaces for the MRI training pipeline.

The package exposes stable, high-level augmentation entry points while
keeping internal implementation details inside their respective modules.

Public API
----------

Training / validation:
    create_train_transforms
    create_val_transforms

Batch-level:
    mixup_batch

Volume-level:
    random_slice_dropout

MRI-specific:
    multiplicative_bias_field

NumPy intensity helpers and low-level spatial helpers remain available from
their individual modules and are intentionally not re-exported here.
"""

from __future__ import annotations

from .factory import (
    create_train_transforms,
    create_val_transforms,
)
from .medical import (
    multiplicative_bias_field,
)
from .mixup import (
    mixup_batch,
)
from .three_d import (
    random_slice_dropout,
)


__all__ = [
    "create_train_transforms",
    "create_val_transforms",
    "mixup_batch",
    "random_slice_dropout",
    "multiplicative_bias_field",
]

