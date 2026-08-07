"""
Grad-CAM (Gradient-weighted Class Activation Mapping)

WHY it exists:
Medical AI cannot be a black box. If the model predicts "ACL Tear",
the radiologist must see WHY. Grad-CAM visualizes where the network is looking
by taking the gradient of the target class score with respect to the final convolutional
feature map. Regions with high positive gradients are "important" for that class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: The neural network
            target_layer: The specific layer to compute CAM on (e.g., model.layer4[-1].conv3)
        """
        self.model = model
        self.target_layer = target_layer
        
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generates the CAM heatmap for a specific class.
        
        Args:
            x: Input image tensor [1, C, H, W]
            target_class: Index of the class to visualize
            
        Returns:
            heatmap: NumPy array [H, W] normalized between 0 and 1
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(x)
        
        if self.activations is None:
            raise RuntimeError("Forward hook did not save activations. Check target layer.")
            
        # Target for backprop
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        if self.gradients is None:
            raise RuntimeError("Backward hook did not save gradients. Check target layer.")
            
        # 1. Global average pooling of gradients
        # alpha_k^c = 1/Z sum_i sum_j (del Y^c / del A^k_ij)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 2. Weight activations by gradients
        activations = self.activations[0] # [C, H, W]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # 3. Sum across channels and apply ReLU
        # L^c = ReLU(sum_k alpha_k^c A^k)
        heatmap = torch.sum(activations, dim=0).squeeze()
        heatmap = F.relu(heatmap)
        
        # 4. Normalize to [0, 1] and resize to original image size
        heatmap = heatmap.detach().cpu().numpy()
        
        if np.max(heatmap) == 0:
            return heatmap
            
        heatmap = heatmap / np.max(heatmap)
        heatmap = cv2.resize(heatmap, (x.shape[-1], x.shape[-2]))
        
        return heatmap
