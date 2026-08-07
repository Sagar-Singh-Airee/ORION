"""Aggregate ternary labelling functions with a deterministic no-Snorkel fallback."""
from __future__ import annotations

import numpy as np

try:
    from snorkel.labeling.model import LabelModel
except ImportError:  # pragma: no cover
    LabelModel = None  # type: ignore[assignment]


def train_label_model(L: np.ndarray, num_classes: int = 2, seed: int = 42) -> np.ndarray:
    """Return ``[P(negative), P(positive)]`` from a `(samples, LFs)` ternary matrix.

    The fallback is intentionally transparent: it averages non-abstaining LF votes
    with Laplace smoothing. It is safer than silently making all abstentions negative
    and keeps the project runnable without an optional Snorkel installation.
    """
    L = np.asarray(L)
    if num_classes != 2 or L.ndim != 2 or not np.isin(L, (-1, 0, 1)).all():
        raise ValueError("L must be a two-class (N, M) matrix containing only -1, 0, 1")
    if LabelModel is not None:
        model = LabelModel(cardinality=2, verbose=False)
        model.fit(L_train=L, n_epochs=500, log_freq=0, seed=seed)
        return model.predict_proba(L)
    observed = L >= 0
    positive_votes = (L == 1).sum(axis=1)
    vote_counts = observed.sum(axis=1)
    # A completely abstaining sample remains explicitly uninformative (0.5/0.5).
    positive = (positive_votes + 1.0) / (vote_counts + 2.0)
    return np.column_stack((1 - positive, positive)).astype(np.float32)


def aggregate_multilabel(label_matrices: np.ndarray) -> np.ndarray:
    """Aggregate `L[:, class, lf]` matrices into classwise positive probabilities."""
    label_matrices = np.asarray(label_matrices)
    if label_matrices.ndim != 3:
        raise ValueError("Expected (N, C, M) label-function matrix")
    return np.column_stack([train_label_model(label_matrices[:, label])[:, 1] for label in range(label_matrices.shape[1])])
