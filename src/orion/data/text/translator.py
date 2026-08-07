"""
Multilingual Report Translator

WHY it exists:
The RSNA dataset contains reports in 9 different languages.
While XLM-RoBERTa handles multilingual text natively, rule-based label extraction
and weak supervision models (like Snorkel) often require English.
This utility standardizes reports to English.
"""

from typing import Optional
from loguru import logger

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

def detect_language(text: str) -> str:
    """
    Detects the language of the text.
    Returns ISO 639-1 language code (e.g., 'en', 'es', 'pt').
    """
    try:
        from langdetect import detect
        return detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en" # Fallback to English

def translate_to_english(text: str, source_lang: str = "auto") -> str:
    """
    Translates text to English.
    """
    if not text.strip():
        return text
        
    if source_lang == "en":
        return text
        
    if not TRANSLATOR_AVAILABLE:
        logger.error("deep-translator not installed. Cannot translate.")
        return text
        
    try:
        translator = GoogleTranslator(source=source_lang, target="en")
        translated = translator.translate(text)
        return translated if translated else text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text
