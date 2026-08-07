"""Hugging Face text encoders returning token-level embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import TEXT_ENCODERS

try:
    from transformers import AutoConfig, AutoModel
except ImportError:  # pragma: no cover
    AutoConfig = AutoModel = None  # type: ignore[assignment]


class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, dropout: float = 0.0, **_: object):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required for text encoders")
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.out_channels = int(self.model.config.hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object) -> torch.Tensor:
        return self.dropout(self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state)


@TEXT_ENCODERS.register("xlm_roberta_base")
class XLMRobertaBase(HuggingFaceTextEncoder):
    def __init__(self, **kwargs: object):
        super().__init__("xlm-roberta-base", **kwargs)
