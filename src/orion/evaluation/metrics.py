"""
Evaluation Metrics

WHY it exists:
The Kaggle metric is Macro-Averaged AUC ROC across 12 targets.
We need a fast, reliable way to compute this metric on validation sets,
ignoring classes that might not have any positive examples in a specific batch.
"""

from typing import Dict, List, Tuple
import numpy as np
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
from loguru import logger

def calculate_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    """
    Computes per-class AUC and Macro AUC.
    
    Args:
        y_true: Ground truth binary labels [N_samples, N_classes]
        y_pred_probs: Predicted probabilities [N_samples, N_classes]
        label_names: List of class names
        
    Returns:
        Dictionary of metrics including 'macro_auc' and per-class aucs.
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn not installed. Cannot compute AUC.")
        return {"macro_auc": 0.0}
        
    metrics = {}
    valid_aucs = []
    
    for i, name in enumerate(label_names):
        true_i = y_true[:, i]
        pred_i = y_pred_probs[:, i]
        
        # ROC AUC requires at least one positive and one negative example
        if len(np.unique(true_i)) == 2:
            auc = roc_auc_score(true_i, pred_i)
            metrics[f"auc_{name}"] = auc
            valid_aucs.append(auc)
        else:
            # If a class is entirely missing from the validation fold (rare but possible),
            # we don't include it in the macro average to avoid NaN.
            metrics[f"auc_{name}"] = float('nan')
            logger.warning(f"Class {name} has only one label type in this batch/fold. AUC undefined.")
            
    if valid_aucs:
        metrics["macro_auc"] = float(np.mean(valid_aucs))
    else:
        metrics["macro_auc"] = 0.0
        
    return metrics
