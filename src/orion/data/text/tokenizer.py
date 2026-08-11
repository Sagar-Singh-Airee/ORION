"""
Radiology Report Tokenizer
==========================

WHY IT EXISTS
-------------

Transformer text encoders do not consume raw strings directly. They require
token IDs, attention masks, and model-specific special tokens.

This module provides one small, inspectable interface for converting
radiology reports into transformer-ready tensors.

Pipeline:

    raw / translated report
            ↓
      report_parser.py
            ↓
      selected report text
            ↓
       ReportTokenizer
            ↓
    input_ids + attention_mask
            ↓
        text encoder

DESIGN
------

The tokenizer intentionally does NOT perform:

    - language detection
    - translation
    - report parsing
    - diagnosis extraction
    - label generation

Those responsibilities belong to the other modules.

The tokenizer only handles model-compatible text encoding.

DEFAULT MODEL
-------------

The default model is XLM-RoBERTa because the project may encounter
multilingual reports and XLM-R provides multilingual subword tokenization.

The tokenizer itself is model-agnostic: any Hugging Face tokenizer compatible
with AutoTokenizer can be supplied through `model_name`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from loguru import logger

try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerBase,
    )

    TRANSFORMERS_AVAILABLE = True

except ImportError:  # pragma: no cover
    AutoTokenizer = None
    PreTrainedTokenizerBase = Any
    TRANSFORMERS_AVAILABLE = False


__all__ = [
    "ReportTokenizer",
]


# ---------------------------------------------------------------------------
# Report tokenizer
# ---------------------------------------------------------------------------


class ReportTokenizer:
    """
    Hugging Face tokenizer wrapper for radiology reports.

    Parameters
    ----------
    model_name:
        Hugging Face tokenizer name or local tokenizer directory.

    max_length:
        Maximum number of tokens retained per report.

    padding:
        Padding strategy.

        "max_length"
            Every returned sequence has exactly max_length tokens.

        "longest"
            Batch is padded only to the longest sequence in that batch.

        False
            No padding is applied.

        The default is "longest" because it avoids unnecessary padding during
        batch inference/training.

    Notes
    -----
    The public call interface remains:

        tokenizer("report")

        tokenizer(["report 1", "report 2"])

    and returns the normal Hugging Face BatchEncoding object.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_length: int = 512,
        padding: str | bool = "longest",
    ) -> None:

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for report tokenization. "
                "Install it with: pip install transformers"
            )

        if not isinstance(
            max_length,
            int,
        ):
            raise TypeError(
                "max_length must be an integer, "
                f"got {type(max_length).__name__}"
            )

        if max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {max_length}"
            )

        if padding not in (
            "max_length",
            "longest",
            False,
            True,
        ):
            raise ValueError(
                "padding must be one of "
                "'max_length', 'longest', True, or False; "
                f"got {padding!r}"
            )

        # Hugging Face treats padding=True as "longest".
        if padding is True:
            padding = "longest"

        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding

        logger.info(
            "Loading report tokenizer: "
            f"{model_name}"
        )

        self.tokenizer: PreTrainedTokenizerBase = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        # Some transformer tokenizers legitimately have no padding token.
        # This wrapper should fail clearly rather than producing a confusing
        # error later during batch encoding.
        if (
            self.padding is not False
            and self.tokenizer.pad_token_id is None
        ):
            raise ValueError(
                f"Tokenizer {model_name!r} has no pad_token_id, "
                "but padding was requested."
            )

    # -------------------------------------------------------------------
    # Input normalization
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize_texts(
        texts: str | Sequence[str],
    ) -> tuple[list[str], bool]:
        """
        Normalize one string or a sequence of strings.

        Returns
        -------
        texts:
            Normalized list of strings.

        was_single:
            Whether the original input was a single string.

        Empty strings are retained rather than silently removed because
        removing an item would change batch alignment.
        """

        if isinstance(
            texts,
            str,
        ):
            return [texts], True

        if not isinstance(
            texts,
            Sequence,
        ):
            raise TypeError(
                "texts must be a string or a sequence of strings, "
                f"got {type(texts).__name__}"
            )

        normalized = list(texts)

        for index, text in enumerate(
            normalized
        ):
            if not isinstance(
                text,
                str,
            ):
                raise TypeError(
                    "Every text item must be a string; "
                    f"item {index} is "
                    f"{type(text).__name__}"
                )

        return normalized, False

    # -------------------------------------------------------------------
    # Tokenization
    # -------------------------------------------------------------------

    def __call__(
        self,
        texts: str | Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize one or more radiology reports.

        Parameters
        ----------
        texts:
            One report string or a sequence of report strings.

        Returns
        -------
        dict[str, torch.Tensor]
            Transformer inputs, normally containing:

                input_ids
                attention_mask

            Additional tokenizer-specific fields are preserved if the
            underlying tokenizer produces them.

        Notes
        -----
        Truncation is always enabled because radiology reports can exceed
        the model's configured context length.
        """

        normalized, _ = (
            self._normalize_texts(
                texts
            )
        )

        if not normalized:
            # Hugging Face tokenizers generally support empty batches poorly
            # across model/tokenizer versions. Return correctly-shaped empty
            # tensors instead of invoking undefined behavior.
            return {
                "input_ids": torch.empty(
                    (0, 0),
                    dtype=torch.long,
                ),
                "attention_mask": torch.empty(
                    (0, 0),
                    dtype=torch.long,
                ),
            }

        encoded = self.tokenizer(
            normalized,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        return {
            key: value
            for key, value in encoded.items()
            if isinstance(
                value,
                torch.Tensor,
            )
        }

    # -------------------------------------------------------------------
    # Explicit batch API
    # -------------------------------------------------------------------

    def encode_batch(
        self,
        texts: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """
        Explicit batch-tokenization alias.

        This exists mainly to make training code self-documenting:

            tokenizer.encode_batch(reports)

        It uses exactly the same behavior as __call__.
        """

        if isinstance(
            texts,
            str,
        ):
            raise TypeError(
                "encode_batch expects a sequence of strings; "
                "use tokenizer(text) for a single report."
            )

        return self(
            texts
        )

    # -------------------------------------------------------------------
    # Single-report API
    # -------------------------------------------------------------------

    def encode_one(
        self,
        text: str,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize exactly one report.

        The returned tensors retain a batch dimension of one.
        """

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "text must be a string"
            )

        return self(
            text
        )

    # -------------------------------------------------------------------
    # Length inspection
    # -------------------------------------------------------------------

    def token_length(
        self,
        text: str,
    ) -> int:
        """
        Return the number of tokens produced before padding.

        This is useful during EDA for determining whether the chosen
        max_length is unnecessarily truncating reports.

        Special tokens are included because they consume model context.
        """

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "text must be a string"
            )

        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_tensors=None,
        )

        input_ids = encoded.get(
            "input_ids"
        )

        if input_ids is None:
            raise RuntimeError(
                "Tokenizer did not return input_ids"
            )

        return len(input_ids)

    # -------------------------------------------------------------------
    # Truncation inspection
    # -------------------------------------------------------------------

    def was_truncated(
        self,
        text: str,
    ) -> bool:
        """
        Return True when the report exceeds max_length.

        This performs tokenization without truncation so that the original
        token count can be compared with the configured limit.
        """

        return (
            self.token_length(text)
            > self.max_length
        )

    # -------------------------------------------------------------------
    # Model/device helper
    # -------------------------------------------------------------------

    @staticmethod
    def move_to_device(
        encoded: dict[str, torch.Tensor],
        device: str | torch.device,
    ) -> dict[str, torch.Tensor]:
        """
        Move tokenized tensors to a model device.

        Keeping device movement outside __call__ is intentional: the same
        tokenizer can therefore be used for CPU preprocessing and later
        moved to CUDA inside the training/inference loop.
        """

        target = torch.device(
            device
        )

        return {
            key: value.to(target)
            for key, value in encoded.items()
        }

