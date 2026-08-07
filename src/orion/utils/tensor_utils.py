"""
Tensor Manipulation Utilities

WHY it exists:
Centralizes common PyTorch operations (padding, truncating, masking, mixed precision casting)
that are repeated across datasets and models. Using standard utility functions prevents bugs
like in-place modification errors or tensor device mismatches.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

def pad_or_truncate_3d(tensor: torch.Tensor, target_slices: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Ensures a 3D volume (or sequence of 2D slices) has exactly `target_slices` along dimension 0.
    
    Args:
        tensor: Shape (S, C, H, W) or (S, H, W) where S is the slice dimension.
        target_slices: Target number of slices.
        pad_value: Value to use for padding.
        
    Returns:
        Tensor of shape (target_slices, ...)
    """
    current_slices = tensor.shape[0]
    
    if current_slices == target_slices:
        return tensor
    elif current_slices > target_slices:
        # Truncate: simple center crop along slice dimension
        start = (current_slices - target_slices) // 2
        return tensor[start : start + target_slices]
    else:
        # Pad: add zero slices (usually at the end)
        pad_size = target_slices - current_slices
        # F.pad format for N-D: (padding_left, padding_right, padding_top, padding_bottom, ...)
        # We want to pad the 0-th dimension.
        # For (S, C, H, W), F.pad takes (W_l, W_r, H_t, H_b, C_front, C_back, S_front, S_back)
        
        dims = tensor.dim()
        # Create pad list of 0s, length = 2 * dims
        pad_tuple = [0] * (2 * dims)
        # We only pad the "back" of the slice dimension (last in PyTorch's reverse pad order)
        # slice dim is dim 0. So it's the last two elements of the pad tuple.
        pad_tuple[-1] = pad_size # Pad after
        
        return F.pad(tensor, pad_tuple, value=pad_value) # type: ignore


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Performs MixUp augmentation.
    MixUp combines two images and their labels linearly.
    
    WHY: Acts as a strong regularizer and encourages the model to behave linearly
    in between training examples, reducing overconfidence.
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def get_device() -> torch.device:
    """Returns the optimal device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
