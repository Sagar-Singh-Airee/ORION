"""
Multimodal Dataset (MRI + Radiology Reports)

WHY it exists:
To train fusion models, we must fetch both the image volume and the corresponding
radiology report text. This dataset wraps the KneeMRIDataset and the ReportTokenizer.
"""

from typing import Dict, Any
import torch

from .knee_mri import KneeMRIDataset
from ..text.tokenizer import ReportTokenizer
from loguru import logger

class MultimodalKneeDataset(KneeMRIDataset):
    def __init__(self, config: Any, split: str = "train", transform: Any | None = None, records: list[dict[str, Any]] | None = None):
        super().__init__(config, split, transform=transform, records=records)
        
        # Initialize text tokenizer
        text_cfg = config.model.text_encoder
        model_name = text_cfg.name if text_cfg else "xlm-roberta-base"
        max_length = text_cfg.max_length if text_cfg else 512
        
        self.tokenizer = ReportTokenizer(model_name=model_name, max_length=max_length)
        logger.info(f"Initialized MultimodalKneeDataset with text encoder: {model_name}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Get the image and labels from the parent class
        data = super().__getitem__(idx)
        
        # 2. Get the text report
        record = self.data_records[idx]
        
        # In a real scenario, this would be `record["report_text"]`
        report_text = record.get("report_text", "No report available.")
        
        # 3. Tokenize
        tokens = self.tokenizer(report_text)
        
        # Squeeze the batch dimension added by the tokenizer (since this is a single item)
        data["input_ids"] = tokens["input_ids"].squeeze(0)
        data["attention_mask"] = tokens["attention_mask"].squeeze(0)
        
        return data
