"""
Text Tokenizer Utility

WHY it exists:
Transformers don't process strings; they process integer token IDs.
This wraps HuggingFace tokenizers to provide consistent padding, truncation,
and formatting for our multimodal architectures.
"""

from typing import Dict, Any, List
import torch
from loguru import logger

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class ReportTokenizer:
    def __init__(self, model_name: str = "xlm-roberta-base", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for tokenization.")
            
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def __call__(self, texts: str | List[str]) -> Dict[str, torch.Tensor]:
        """
        Tokenizes text with padding and truncation.
        Returns a dictionary with 'input_ids' and 'attention_mask'.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded
