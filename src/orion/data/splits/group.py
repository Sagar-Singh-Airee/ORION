"""
Cross-Validation Splitting Strategies

WHY it exists:
In medical imaging, multiple studies can belong to the SAME patient.
If we use standard K-Fold or random splitting, Patient A's study from 2024 might
end up in the training set, while Patient A's study from 2025 ends up in the test set.
This is DATA LEAKAGE. The model might just learn Patient A's anatomy instead of the disease.
We must use GroupKFold, where the "group" is the PatientID.
"""

from typing import Tuple, List
import numpy as np

try:
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    
from loguru import logger

def get_group_kfold_splits(
    X: np.ndarray, 
    y: np.ndarray, 
    groups: np.ndarray, 
    n_splits: int = 5, 
    stratified: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generates cross-validation splits grouped by patient ID.
    
    Args:
        X: Feature array or indices
        y: Labels array
        groups: Array of Patient IDs corresponding to each sample
        n_splits: Number of folds
        stratified: If True, attempts to balance the class distribution across folds.
                    (StratifiedGroupKFold requires newer scikit-learn).
                    
    Returns:
        List of (train_idx, val_idx) tuples.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for advanced cross-validation splitting.")
        
    if stratified:
        logger.info(f"Using StratifiedGroupKFold with {n_splits} splits.")
        cv = StratifiedGroupKFold(n_splits=n_splits)
    else:
        logger.info(f"Using standard GroupKFold with {n_splits} splits.")
        cv = GroupKFold(n_splits=n_splits)
        
    splits = list(cv.split(X, y, groups=groups))
    return splits
