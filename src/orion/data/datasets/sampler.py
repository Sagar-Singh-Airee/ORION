"""
Custom Data Samplers

WHY it exists:
Standard PyTorch RandomSampler just shuffles indices. In medical imaging, we often
have severe class imbalance (e.g., 5000 normal ACLs vs 500 torn ACLs). 
A WeightedRandomSampler ensures the network sees rare classes more frequently.
"""

from typing import List, Sequence
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from loguru import logger

def create_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Creates a PyTorch WeightedRandomSampler to balance multi-label datasets.
    
    Args:
        labels: NumPy array of shape (N_samples, N_classes)
        
    WHY: Multi-label balancing is tricky. If we sample purely for a rare fracture,
    we might over-sample the effusion it's correlated with.
    A common heuristic is to assign each sample a weight based on its rarest positive class.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must have shape (N, C)")
    N, C = labels.shape
    
    # 1. Calculate frequency of each class
    known = labels >= 0
    class_counts = np.where(known, labels, 0).sum(axis=0)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # 2. Calculate weight for each class (inverse frequency)
    class_weights = N / (C * class_counts)
    
    # 3. Assign weight to each sample
    sample_weights = np.zeros(N)
    
    for i in range(N):
        # Find which classes are positive in this sample
        pos_classes = np.where(labels[i] > 0.5)[0]
        
        if len(pos_classes) > 0:
            # Assign the weight of the RAREST class present in the sample
            sample_weights[i] = np.max(class_weights[pos_classes])
        else:
            # If all negative, assign the minimum class weight
            sample_weights[i] = np.min(class_weights)
            
    logger.info(f"Created balanced sampler. Weight range: [{np.min(sample_weights):.2f}, {np.max(sample_weights):.2f}]")
    
    # Create the PyTorch sampler
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), # type: ignore
        num_samples=N,
        replacement=True
    )
