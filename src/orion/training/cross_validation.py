"""Persist reproducible, patient-safe fold assignments before training starts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.splits.stratified import iterative_group_kfold


def make_fold_assignments(frame: pd.DataFrame, group_column: str, label_columns: list[str], n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    if group_column not in frame or any(column not in frame for column in label_columns):
        raise ValueError("Missing group or label columns")
    output = frame.copy(); output["fold"] = -1
    labels = output[label_columns].to_numpy()
    for fold, (_, validation) in enumerate(iterative_group_kfold(labels, output[group_column].to_numpy(), n_splits, seed)):
        output.iloc[validation, output.columns.get_loc("fold")] = fold
    if (output["fold"] < 0).any(): raise RuntimeError("Some rows were not assigned a fold")
    return output


def save_fold_assignments(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(output_path, index=False); return output_path
