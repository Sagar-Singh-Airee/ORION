"""
Programmatic Weak Supervision (Snorkel Label Model)

WHY it exists:
The dataset only has 58 expert-labeled studies, but 4,900+ radiology reports.
Instead of treating our NLP rules (from label_extractor.py) as absolute ground truth,
we can write multiple "noisy" labeling functions (LFs):
  - LF1: Keyword matching
  - LF2: NegEx regex
  - LF3: Zero-shot LLM prediction
  
Snorkel's LabelModel learns the hidden accuracy of each rule by observing how they
agree/disagree, and aggregates them into a single, highly accurate probabilistic label.
This turns 4,900 unlabelled studies into 4,900 weakly-supervised training targets.
"""

from typing import List, Dict
import numpy as np
from loguru import logger

try:
    from snorkel.labeling.model import LabelModel
    SNORKEL_AVAILABLE = True
except ImportError:
    SNORKEL_AVAILABLE = False

def train_label_model(L: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Trains a Snorkel LabelModel on a matrix of heuristic labels.
    
    Args:
        L: Label matrix of shape (N_samples, N_lfs) where each entry is
           0 (Negative), 1 (Positive), or -1 (Abstain).
           
    Returns:
        Y_probs: Array of shape (N_samples, num_classes) containing 
                 probabilistic labels (e.g., 0.85 probability of Positive).
    """
    if not SNORKEL_AVAILABLE:
        raise ImportError("Snorkel is required. Install via `pip install snorkel`.")
        
    logger.info(f"Training Snorkel LabelModel on matrix of shape {L.shape}")
    
    label_model = LabelModel(cardinality=num_classes, verbose=True)
    
    # Train the generative model to learn the accuracies of the LFs
    label_model.fit(L_train=L, n_epochs=500, log_freq=100, seed=42)
    
    # Predict probabilistic labels
    Y_probs = label_model.predict_proba(L)
    
    return Y_probs
