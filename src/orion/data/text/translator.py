
"""
Radiology Report Translator
===========================

WHY IT EXISTS
-------------

The report-only portion of the dataset may contain reports written in
multiple languages.

The weak-label extractor operates on English text, so multilingual reports
are translated to English before section parsing and weak-label extraction.

Pipeline:

    raw report
        ↓
    language detection
        ↓
    NLLB translation
        ↓
    English report
        ↓
    report_parser.py
        ↓
    label_extractor.py

KAGGLE / OFFLINE DESIGN
-----------------------

Competition inference environments may have internet disabled.

Therefore:

    local_path != None
        → load an already-attached local NLLB model

    local_path == None
        → load facebook/nllb-200-distilled-600M from Hugging Face

The second mode is useful during training/EDA but should not be relied on
for an offline inference kernel.

CACHING
-------

Translation is expensive.

Every translation is cached using:

    SHA256(source_language + source_text)

This means:

    same text + same source language
        → same cache entry

Changing either invalidates the cache.

IMPORTANT
---------

This module performs translation only.

It does not parse radiology sections and does not generate medical labels.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Final

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


try:
    from langdetect import DetectorFactory
    from langdetect import detect as _langdetect

    # Make language detection deterministic.
    DetectorFactory.seed = 0

except ImportError:  # pragma: no cover
    _langdetect = None


__all__ = [
    "ISO_TO_FLORES",
    "TARGET_FLORES",
    "ReportTranslator",
]


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# ISO 639-1 / langdetect language codes → NLLB FLORES-200 language codes.
#
# Keep this list aligned with the languages actually expected in the dataset.
# Unsupported detected languages are handled explicitly rather than silently
# pretending they are English.
ISO_TO_FLORES: Final[dict[str, str]] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "zh-cn": "zho_Hans",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "hi": "hin_Deva",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
}


TARGET_FLORES: Final[str] = "eng_Latn"

_DEFAULT_MODEL_ID: Final[str] = (
    "facebook/nllb-200-distilled-600M"
)


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class ReportTranslator:
    """
    Translate multilingual radiology reports into English.

    Parameters
    ----------
    local_path:
        Local NLLB model directory.

        Use this for offline Kaggle inference.

        If None, the Hugging Face model ID is used.

    cache_dir:
        Directory used for persistent translation cache.

    device:
        Explicit device such as "cuda" or "cpu".
        If omitted, CUDA is selected when available.

    batch_size:
        Number of reports translated together.

    max_length:
        Maximum sequence length used by the tokenizer and generation.

    Notes
    -----
    The public API intentionally remains compatible with the original
    implementation:

        ReportTranslator(...)
        detect_language(...)
        translate(...)
        translate_batch(...)
    """

    def __init__(
        self,
        local_path: str | Path | None,
        cache_dir: str | Path,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:

        if _langdetect is None:
            raise ImportError(
                "langdetect is required for multilingual report "
                "translation. Install it with: pip install langdetect"
            )

        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )

        if max_length <= 0:
            raise ValueError(
                f"max_length must be positive, got {max_length}"
            )

        # ---------------------------------------------------------------
        # Device
        # ---------------------------------------------------------------

        requested_device = (
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        if requested_device.startswith(
            "cuda"
        ) and not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is not available"
            )

        self.device = torch.device(
            requested_device
        )

        self.batch_size = int(
            batch_size
        )

        self.max_length = int(
            max_length
        )

        # ---------------------------------------------------------------
        # Model source
        # ---------------------------------------------------------------

        if local_path is not None:

            model_path = Path(
                local_path
            ).expanduser()

            if not model_path.exists():
                raise FileNotFoundError(
                    "Local translation model path does not exist: "
                    f"{model_path}"
                )

            if not model_path.is_dir():
                raise ValueError(
                    "local_path must point to a model directory: "
                    f"{model_path}"
                )

            model_source = str(
                model_path
            )

        else:
            model_source = _DEFAULT_MODEL_ID

        self.model_source = model_source

        # ---------------------------------------------------------------
        # Load tokenizer/model
        # ---------------------------------------------------------------

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_source
            )
        )

        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_source
            )
            .to(self.device)
            .eval()
        )

        # Translation inference does not need gradients.
        self.model.requires_grad_(
            False
        )

        # ---------------------------------------------------------------
        # Cache
        # ---------------------------------------------------------------

        self.cache_dir = (
            Path(cache_dir).expanduser()
            / "translations"
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------

    @staticmethod
    def _cache_key(
        text: str,
        src_lang: str,
    ) -> str:
        """
        Generate a deterministic cache key.

        The source language is part of the key because identical text can
        represent different strings/meanings in different languages.
        """

        payload = (
            f"{src_lang}\x00{text}"
        ).encode("utf-8")

        return hashlib.sha256(
            payload
        ).hexdigest()

    def _cache_path(
        self,
        key: str,
    ) -> Path:
        """Return the JSON cache path for one translation."""

        return (
            self.cache_dir
            / f"{key}.json"
        )

    def _read_cache(
        self,
        path: Path,
    ) -> str | None:
        """
        Read a cached translation safely.

        Corrupt/incomplete cache files are treated as cache misses instead
        of crashing the entire dataset-processing run.
        """

        try:

            payload = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            translation = payload.get(
                "translation"
            )

            if not isinstance(
                translation,
                str,
            ):
                return None

            return translation

        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ):
            return None

    def _write_cache(
        self,
        path: Path,
        *,
        src_lang: str,
        original: str,
        translation: str,
    ) -> None:
        """
        Atomically write one translation cache entry.

        A temporary file prevents an interrupted process from leaving a
        partially-written JSON file that looks like a valid cache entry.
        """

        payload = {
            "src_lang": src_lang,
            "original": original,
            "translation": translation,
        }

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fd: int | None = None
        temporary_path: str | None = None

        try:

            fd, temporary_path = (
                tempfile.mkstemp(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    text=True,
                )
            )

            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as handle:

                fd = None

                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                )

                handle.write("\n")
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

            os.replace(
                temporary_path,
                path,
            )

            temporary_path = None

        finally:

            if fd is not None:
                os.close(fd)

            if temporary_path is not None:
                try:
                    os.unlink(
                        temporary_path
                    )
                except OSError:
                    pass

    # -------------------------------------------------------------------
    # Language detection
    # -------------------------------------------------------------------

    def detect_language(
        self,
        text: str,
    ) -> str:
        """
        Detect the source language and return its NLLB FLORES code.

        Returns English for:

        - empty text
        - detection failures
        - unsupported language codes

        This preserves the original public behavior while keeping the
        fallback explicit and centralized.

        IMPORTANT
        ---------

        Language detection on very short strings is inherently unreliable.
        Radiology reports are generally long enough for detection, but very
        short inputs should not be over-interpreted.
        """

        if not isinstance(
            text,
            str,
        ):
            return TARGET_FLORES

        text = text.strip()

        if not text:
            return TARGET_FLORES

        try:
            iso = _langdetect(
                text
            )
        except Exception:
            return TARGET_FLORES

        return ISO_TO_FLORES.get(
            iso,
            TARGET_FLORES,
        )

    # -------------------------------------------------------------------
    # Translation
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def _translate_batch(
        self,
        texts: list[str],
        src_flores: str,
    ) -> list[str]:
        """
        Translate one batch from one source language to English.

        All texts in the batch must use the same source FLORES language
        because NLLB's tokenizer source language is batch-level state.
        """

        if not texts:
            return []

        if src_flores == TARGET_FLORES:
            return list(texts)

        if src_flores not in set(
            ISO_TO_FLORES.values()
        ):
            raise ValueError(
                "Unsupported source FLORES language: "
                f"{src_flores!r}"
            )

        self.tokenizer.src_lang = (
            src_flores
        )

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        encoded = {
            key: value.to(
                self.device
            )
            for key, value in encoded.items()
        }

        forced_bos = (
            self.tokenizer.convert_tokens_to_ids(
                TARGET_FLORES
            )
        )

        if (
            forced_bos is None
            or forced_bos == self.tokenizer.unk_token_id
        ):
            raise RuntimeError(
                "Could not resolve NLLB target language "
                f"token {TARGET_FLORES!r}"
            )

        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=forced_bos,
            max_length=self.max_length,
            num_beams=4,
            do_sample=False,
        )

        translations = (
            self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

        if len(translations) != len(
            texts
        ):
            raise RuntimeError(
                "Translation model returned "
                f"{len(translations)} outputs for "
                f"{len(texts)} inputs"
            )

        return [
            translation.strip()
            for translation in translations
        ]

    # -------------------------------------------------------------------
    # Single report
    # -------------------------------------------------------------------

    def translate(
        self,
        text: str,
        src_lang: str | None = None,
    ) -> str:
        """
        Translate one report into English.

        Parameters
        ----------
        text:
            Source report.

        src_lang:
            Optional NLLB FLORES source language, e.g. "fra_Latn".

            If omitted, language detection is performed.

        Returns
        -------
        str
            English translation.

        Empty input is returned unchanged.
        """

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            return text

        text = text.strip()

        if src_lang is None:
            source = self.detect_language(
                text
            )
        else:
            source = str(
                src_lang
            ).strip()

            if not source:
                source = self.detect_language(
                    text
                )

        key = self._cache_key(
            text,
            source,
        )

        cache_path = self._cache_path(
            key
        )

        cached = self._read_cache(
            cache_path
        )

        if cached is not None:
            return cached

        if source == TARGET_FLORES:
            translation = text

        else:
            translation = self._translate_batch(
                [text],
                source,
            )[0]

        self._write_cache(
            cache_path,
            src_lang=source,
            original=text,
            translation=translation,
        )

        return translation

    # -------------------------------------------------------------------
    # Batch translation
    # -------------------------------------------------------------------

    def translate_batch(
        self,
        texts: list[str],
    ) -> list[str]:
        """
        Translate multiple reports efficiently.

        Reports are:

            1. checked for validity
            2. language-detected
            3. resolved from cache when possible
            4. grouped by source language
            5. translated in batches
            6. written back to cache

        Original input order is preserved exactly.
        """

        if not isinstance(
            texts,
            list,
        ):
            raise TypeError(
                "texts must be a list[str]"
            )

        if not texts:
            return []

        for index, text in enumerate(
            texts
        ):
            if not isinstance(
                text,
                str,
            ):
                raise TypeError(
                    "Every item in texts must be a string; "
                    f"item {index} is {type(text).__name__}"
                )

        results: list[
            str | None
        ] = [None] * len(texts)

        # source language → indices that require translation.
        by_language: dict[
            str,
            list[int],
        ] = {}

        # ---------------------------------------------------------------
        # Resolve empty reports and cache hits first.
        # ---------------------------------------------------------------

        for index, raw_text in enumerate(
            texts
        ):

            text = raw_text.strip()

            if not text:
                results[index] = raw_text
                continue

            source = self.detect_language(
                text
            )

            key = self._cache_key(
                text,
                source,
            )

            cache_path = self._cache_path(
                key
            )

            cached = self._read_cache(
                cache_path
            )

            if cached is not None:
                results[index] = cached
                continue

            if source == TARGET_FLORES:
                # No model call is necessary.
                results[index] = text

                self._write_cache(
                    cache_path,
                    src_lang=source,
                    original=text,
                    translation=text,
                )

                continue

            by_language.setdefault(
                source,
                [],
            ).append(index)

        # ---------------------------------------------------------------
        # Translate uncached reports language-by-language.
        # ---------------------------------------------------------------

        for source, indices in (
            by_language.items()
        ):

            for start in range(
                0,
                len(indices),
                self.batch_size,
            ):

                chunk_indices = indices[
                    start:
                    start + self.batch_size
                ]

                chunk_texts = [
                    texts[index].strip()
                    for index in chunk_indices
                ]

                translations = (
                    self._translate_batch(
                        chunk_texts,
                        source,
                    )
                )

                for (
                    index,
                    original,
                    translation,
                ) in zip(
                    chunk_indices,
                    chunk_texts,
                    translations,
                    strict=True,
                ):

                    results[index] = (
                        translation
                    )

                    key = (
                        self._cache_key(
                            original,
                            source,
                        )
                    )

                    self._write_cache(
                        self._cache_path(
                            key
                        ),
                        src_lang=source,
                        original=original,
                        translation=translation,
                    )

        # ---------------------------------------------------------------
        # Final safety check.
        # ---------------------------------------------------------------

        unresolved = [
            index
            for index, result in enumerate(
                results
            )
            if result is None
        ]

        if unresolved:
            raise RuntimeError(
                "Translation failed to produce results for "
                f"{len(unresolved)} input(s); "
                f"indices={unresolved[:10]}"
            )

        return [
            result
            for result in results
            if result is not None
        ]
