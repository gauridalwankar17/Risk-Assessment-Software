"""
OCR and Translation Module

This module provides:
- Arabic text extraction from images using EasyOCR or Tesseract
- Arabic to English translation using M2M100 or NLLB-200
"""

from .ocr_engine import OCREngine
from .translation_engine import TranslationEngine

__all__ = ["OCREngine", "TranslationEngine"]