"""
Custom Collate Functions

WHY it exists:
PyTorch's default `collate_fn` assumes all tensors in a batch have the exact same shape.
If we use variable-length slice sampling (e.g., keeping all slices without truncation),
default collate will crash. This function handles padding variable-length sequences within a batch.
"""

from typing import List, Dict, Any
import torch
import torch.nn.functional as F

def variable_length_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of dictionaries into a batched dictionary,
    padding images (slice sequences) to the maximum length in the batch.
    """
    elem = batch[0]
    collated = {}
    
    for key in elem.keys():
        if key == "image":
            # Images might have different number of slices
            # Shape: (Slices, Channels, H, W)
            images = [item[key] for item in batch]
            max_slices = max(img.shape[0] for img in images)
            
            padded_images = []
            for img in images:
                pad_size = max_slices - img.shape[0]
                if pad_size > 0:
                    # Pad the slice dimension (dim 0). 
                    # F.pad for (S, C, H, W) is (W, W, H, H, C, C, S, S)
                    padded_img = F.pad(img, (0, 0, 0, 0, 0, 0, 0, pad_size), value=0.0)
                    padded_images.append(padded_img)
                else:
                    padded_images.append(img)
                    
            collated[key] = torch.stack(padded_images, dim=0)
            
            # Create an attention mask for the slice sequence
            # 1 for real slices, 0 for padded slices
            masks = []
            for img in images:
                seq_len = img.shape[0]
                mask = torch.zeros(max_slices, dtype=torch.bool)
                mask[:seq_len] = True
                masks.append(mask)
            collated["slice_mask"] = torch.stack(masks, dim=0)
            
        elif isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
            
        elif isinstance(elem[key], (int, float, str)):
            collated[key] = [item[key] for item in batch]
            
        else:
            collated[key] = [item[key] for item in batch]
            
    return collated
