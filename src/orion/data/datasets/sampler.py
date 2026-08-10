"""
Custom Data Samplers

WHY it exists:
Standard PyTorch RandomSampler just shuffles indices. In medical imaging, we often
have severe class imbalance (e.g., 5000 normal ACLs vs 500 torn ACLs).
A WeightedRandomSampler ensures the network sees rare classes more frequently.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from loguru import logger
from torch.utils.data import WeightedRandomSampler

__all__ = ["create_balanced_sampler"]

# Ternary label convention used throughout this codebase: -1 = abstain (unknown),
# 0 = negative, 1 = positive. Anything strictly above this threshold counts as
# positive; both -1 and 0 fall below it.
_POSITIVE_THRESHOLD = 0.5


def _compute_class_weights(labels: np.ndarray, label_names: Sequence[str] | None) -> np.ndarray:
    """Inverse-frequency weight per class: a class with fewer positive examples gets a larger weight.

    A class with zero positives across the whole split is a plausible sign of a
    label/config mismatch (the same failure mode guarded against when loading
    metadata), so it's flagged rather than silently absorbed into a default weight.
    """
    n_samples, n_classes = labels.shape
    known = labels >= 0
    positive_counts = np.where(known, labels, 0).sum(axis=0)

    zero_positive = np.where(positive_counts == 0)[0]
    if len(zero_positive) > 0:
        names = [label_names[i] if label_names is not None else str(i) for i in zero_positive]
        logger.warning(f"Class(es) with zero positive examples in this split: {names}")

    safe_counts = np.maximum(positive_counts, 1)
    return n_samples / (n_classes * safe_counts)


def _compute_sample_weights(labels: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Per sample: the weight of its rarest positive class, or the global minimum weight if fully negative.

    Vectorized instead of a per-row Python loop — with C=12 (small) and N in the
    thousands, per-row numpy call overhead previously dominated the actual work.
    """
    positive_mask = labels > _POSITIVE_THRESHOLD
    masked_weights = np.where(positive_mask, class_weights[np.newaxis, :], -np.inf)
    row_max = masked_weights.max(axis=1)
    has_positive = positive_mask.any(axis=1)
    return np.where(has_positive, row_max, class_weights.min())


def create_balanced_sampler(
    labels: np.ndarray,
    label_names: Sequence[str] | None = None,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Creates a PyTorch WeightedRandomSampler to balance multi-label datasets.

    Args:
        labels: array of shape (N_samples, N_classes), ternary-encoded
            (-1 = abstain, 0 = negative, 1 = positive).
        label_names: optional class names (e.g. FINDINGS), used only to make the
            zero-positive-class warning name the actual finding instead of an index.
        num_samples: samples to draw per epoch; defaults to N_samples.
        replacement: whether to sample with replacement (required for oversampling
            rare classes beyond their natural count; default True).

    Returns:
        A WeightedRandomSampler ready to pass to a DataLoader.

    WHY: Multi-label balancing is tricky. If we sample purely for a rare fracture,
    we might over-sample the effusion it's correlated with. A common heuristic is
    to assign each sample a weight based on its rarest positive class.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (N, C), got {labels.shape}")
    n_samples, n_classes = labels.shape
    if n_samples == 0:
        raise ValueError("Cannot create a sampler for an empty label array (0 samples)")
    if n_classes == 0:
        raise ValueError("Cannot create a sampler with 0 classes")
    if label_names is not None and len(label_names) != n_classes:
        raise ValueError(f"label_names has {len(label_names)} entries but labels has {n_classes} classes")
    if labels.min() < -1 or labels.max() > 1:
        raise ValueError(f"labels must be in [-1, 1] (-1 = abstain); got range [{labels.min()}, {labels.max()}]")
    if num_samples is not None and num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    class_weights = _compute_class_weights(labels, label_names)
    sample_weights = _compute_sample_weights(labels, class_weights)

    logger.info(
        f"Created balanced sampler for {n_samples} sample(s), {n_classes} class(es). "
        f"Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]"
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=num_samples if num_samples is not None else n_samples,
        replacement=replacement,
    )