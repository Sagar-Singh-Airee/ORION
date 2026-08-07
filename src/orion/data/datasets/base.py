"""
Dataset Base Class

WHY it exists:
Provides common functionality for all PyTorch datasets in this project,
such as parsing configuration, setting up caching, and defining the API contract.
"""

from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from loguru import logger

class BaseDataset(Dataset):
    def __init__(self, config: DictConfig, split: str = "train"):
        """
        Args:
            config: The data configuration from OmegaConf.
            split: 'train', 'val', or 'test'.
        """
        self.config = config
        self.split = split
        
        # Load dataset metadata (e.g., CSV with patient IDs and labels)
        self.data_records = self._load_data_records()
        
        logger.info(f"Initialized {self.__class__.__name__} ({split}): {len(self)} records.")
        
    def _load_data_records(self) -> List[Dict[str, Any]]:
        """
        Subclasses must implement this to load the list of studies/patients.
        Returns a list of dictionaries, where each dict represents one example.
        """
        raise NotImplementedError
        
    def __len__(self) -> int:
        return len(self.data_records)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Subclasses must implement the logic to load a specific item.
        Returns a dict containing 'image', 'label', 'metadata', etc.
        """
        raise NotImplementedError
