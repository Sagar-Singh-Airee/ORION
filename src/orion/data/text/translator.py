"""Translate radiology reports (9 languages) to English before label_extractor.py runs.

Kaggle constraint that shapes this whole file: competition inference kernels typically
run with internet OFF. `facebook/nllb-200-distilled-600M` (~2.4GB) must be attached as a
Kaggle Dataset input and loaded from local path — NOT downloaded via `from_pretrained`
with a hub ID at inference time. Training/EDA sessions (internet on) can pull from hub
directly; pass `local_path=None` there.

Caching is not optional here: translation is the slowest step in the whole pipeline and
Kaggle sessions are ephemeral, so every translated report is cached to disk keyed by a
hash of (source text, source lang) and re-used across reruns within a session.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from langdetect import detect as _langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0  # deterministic detection
except ImportError:  # pragma: no cover - optional dep guard
    _langdetect = None

# ISO 639-1 -> FLORES-200 code, restricted to languages expected in this dataset
# (16 institutions, 5 continents). Extend as EDA reveals the actual language mix.
ISO_TO_FLORES = {
    "en": "eng_Latn", "es": "spa_Latn", "pt": "por_Latn", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "zh-cn": "zho_Hans", "zh": "zho_Hans",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "ru": "rus_Cyrl",
    "hi": "hin_Deva", "nl": "nld_Latn", "pl": "pol_Latn",
}
TARGET_FLORES = "eng_Latn"


class ReportTranslator:
    def __init__(
        self,
        local_path: str | Path | None,
        cache_dir: str | Path,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ):
        """
        local_path: path to a locally-attached model dir (Kaggle Dataset input) for
            internet-off inference. If None, loads from the HF hub (training/EDA only).
        """
        if _langdetect is None:
            raise ImportError("langdetect is required: pip install langdetect")

        model_source = str(local_path) if local_path else "facebook/nllb-200-distilled-600M"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_source).to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length

        self.cache_dir = Path(cache_dir) / "translations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cache_key(text: str, src_lang: str) -> str:
        h = hashlib.sha256(f"{src_lang}\x00{text}".encode("utf-8")).hexdigest()
        return h

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def detect_language(self, text: str) -> str:
        """Returns a FLORES-200 code, defaulting to English on empty/undetectable text."""
        if not text or not text.strip():
            return TARGET_FLORES
        try:
            iso = _langdetect(text)
        except Exception:
            return TARGET_FLORES
        return ISO_TO_FLORES.get(iso, TARGET_FLORES)

    @torch.inference_mode()
    def _translate_batch(self, texts: list[str], src_flores: str) -> list[str]:
        if src_flores == TARGET_FLORES:
            return texts
        self.tokenizer.src_lang = src_flores
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        forced_bos = self.tokenizer.convert_tokens_to_ids(TARGET_FLORES)
        out = self.model.generate(
            **enc, forced_bos_token_id=forced_bos, max_length=self.max_length, num_beams=4,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def translate(self, text: str, src_lang: str | None = None) -> str:
        """Single-report translation with disk cache. src_lang: FLORES code, auto-detected
        if omitted."""
        if not text or not text.strip():
            return text

        src_flores = src_lang or self.detect_language(text)
        key = self._cache_key(text, src_flores)
        cache_path = self._cache_path(key)
        if cache_path.exists():
            return json.loads(cache_path.read_text())["translation"]

        translation = self._translate_batch([text], src_flores)[0] if src_flores != TARGET_FLORES else text
        cache_path.write_text(json.dumps({
            "src_lang": src_flores, "original": text, "translation": translation,
        }))
        return translation

    def translate_batch(self, texts: list[str]) -> list[str]:
        """Batched translation grouped by detected source language for generate() efficiency.
        Cache-hits are resolved without touching the model."""
        n = len(texts)
        results: list[str | None] = [None] * n
        by_lang: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = text
                continue
            src_flores = self.detect_language(text)
            key = self._cache_key(text, src_flores)
            cache_path = self._cache_path(key)
            if cache_path.exists():
                results[i] = json.loads(cache_path.read_text())["translation"]
            else:
                by_lang.setdefault(src_flores, []).append(i)

        for src_flores, idxs in by_lang.items():
            for start in range(0, len(idxs), self.batch_size):
                chunk_idxs = idxs[start:start + self.batch_size]
                chunk_texts = [texts[i] for i in chunk_idxs]
                translations = self._translate_batch(chunk_texts, src_flores)
                for i, orig, trans in zip(chunk_idxs, chunk_texts, translations):
                    results[i] = trans
                    self._cache_path(self._cache_key(orig, src_flores)).write_text(json.dumps({
                        "src_lang": src_flores, "original": orig, "translation": trans,
                    }))

        return results  # type: ignore[return-value]