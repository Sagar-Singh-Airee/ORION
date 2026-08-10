"""
Multimodal Dataset (MRI + Radiology Reports)

WHY it exists:
To train fusion models, we must fetch both the image volume and the corresponding
radiology report text. This dataset wraps the KneeMRIDataset and the ReportTokenizer.
"""
from __future__ import annotations

from typing import Any

import torch
from loguru import logger

from ..text.tokenizer import ReportTokenizer
from .knee_mri import KneeMRIDataset, _first, _get

__all__ = ["MultimodalKneeDataset"]


class MultimodalKneeDataset(KneeMRIDataset):
    """Adds tokenized report text to every KneeMRIDataset sample."""

    def __init__(
        self,
        config: Any,
        split: str = "train",
        transform: Any | None = None,
        records: list[dict[str, Any]] | None = None,
    ):
        super().__init__(config, split, transform=transform, records=records)

        # Same defensive config access as the parent class (KneeMRIDataset's _get/_first):
        # raw `config.model.text_encoder.name` attribute access would raise an unclear
        # error the moment any part of that chain is missing or the config is a plain
        # dict rather than an OmegaConf object.
        model_cfg = _get(config, "model", {})
        text_cfg = _get(model_cfg, "text_encoder", {})
        model_name = _first(text_cfg, "name", "model_name", default="xlm-roberta-base")
        max_length = int(_first(text_cfg, "max_length", default=512))
        if max_length <= 0:
            raise ValueError(f"model.text_encoder.max_length must be positive, got {max_length}")

        self.tokenizer = ReportTokenizer(model_name=model_name, max_length=max_length)
        logger.info(f"Initialized MultimodalKneeDataset (text encoder: {model_name}, max_length: {max_length})")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # KneeMRIDataset.__getitem__ already resolves report_text (defaulting to "" for
        # a missing report, its established contract) and includes it in `data` — no
        # need to re-index self.data_records here, and doing so previously applied a
        # *different*, inconsistent default ("No report available.") that would have
        # been tokenized as if it were real report content.
        data = super().__getitem__(idx)
        tokens = self.tokenizer(data["report_text"])

        for key in ("input_ids", "attention_mask"):
            if key not in tokens:
                raise ValueError(f"Tokenizer output is missing {key!r}; got keys {list(tokens.keys())}")

        input_ids, attention_mask = tokens["input_ids"], tokens["attention_mask"]
        for name, tensor in (("input_ids", input_ids), ("attention_mask", attention_mask)):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != 1:
                # squeeze(0) silently no-ops on a non-1-sized dim rather than raising,
                # which would otherwise let a malformed (B, L) tensor through with B != 1
                # and quietly break variable_length_collate's downstream stacking.
                shape = tuple(getattr(tensor, "shape", ()))
                raise ValueError(f"Expected tokenizer {name} shaped (1, seq_len), got {shape}")

        # Squeeze the batch dimension added by the tokenizer (since this is a single item).
        data["input_ids"] = input_ids.squeeze(0)
        data["attention_mask"] = attention_mask.squeeze(0)
        return data