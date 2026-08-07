"""Strict submission creation to catch ID/label ordering mistakes before upload."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def create_submission(study_ids: Sequence[object], probabilities: np.ndarray, label_names: Sequence[str]) -> pd.DataFrame:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(study_ids), len(label_names)):
        raise ValueError("probabilities must be shaped (len(study_ids), len(label_names))")
    if len(set(map(str, study_ids))) != len(study_ids):
        raise ValueError("Study IDs are not unique")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Submission probabilities must be finite values in [0, 1]")
    return pd.DataFrame(probabilities, columns=list(label_names)).assign(study_id=list(study_ids)).loc[:, ["study_id", *label_names]]


def save_submission(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.7f")
    return path
