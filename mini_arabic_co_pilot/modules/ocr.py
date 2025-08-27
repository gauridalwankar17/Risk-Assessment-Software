import easyocr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from PIL import Image
import io
import numpy as np
from typing import Dict, List, Any
import re

logger = logging.getLogger(__name__)

class OCRTranslator:
    """
    OCR and Translation module for Arabic text extraction and translation
    """
    
    def __init__(self):
        """Initialize OCR reader and translation models"""
        try:
            # Initialize EasyOCR for Arabic and English
            self.reader = easyocr.Reader(['ar', 'en'], gpu=torch.cuda.is_available())
            logger.info("EasyOCR initialized successfully")
            
            # Initialize translation models
            self._init_translation_models()
            
        except Exception as e:
            logger.error(f"Error initializing OCRTranslator: {str(e)}")
            raise
    
    def _init_translation_models(self):
        """Initialize Hugging Face translation models"""
        try:
            # NLLB-200 model for multilingual translation
            model_name = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Language code mapping
            self.language_codes = {
                'en': 'eng_Latn',
                'ar': 'arb_Arab',
                'fr': 'fra_Latn',
                'de': 'deu_Latn',
                'es': 'spa_Latn',
                'zh': 'zho_Hans'
            }
            
            logger.info("Translation models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing translation models: {str(e)}")
            # Fallback to simpler approach
            self.tokenizer = None
            self.model = None
    
    def process_image(self, image_content: bytes, filename: str, target_language: str = "en") -> Dict[str, Any]:
        """
        Process image: extract text and translate
        
        Args:
            image_content: Raw image bytes
            filename: Original filename
            target_language: Target language for translation
            
        Returns:
            Dictionary with extracted text and translation
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_content))
            
            # Extract text using OCR
            extracted_text = self._extract_text(image)
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No text detected in image",
                    "extracted_text": "",
                    "translated_text": "",
                    "confidence": 0.0
                }
            
            # Translate extracted text
            translated_text = self._translate_text(extracted_text, target_language)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(extracted_text)
            
            return {
                "success": True,
                "filename": filename,
                "extracted_text": extracted_text,
                "translated_text": translated_text,
                "source_language": "ar",  # Assuming Arabic input
                "target_language": target_language,
                "confidence": confidence,
                "word_count": len(extracted_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "translated_text": ""
            }
    
    def _extract_text(self, image: Image.Image) -> str:
        """Extract text from image using EasyOCR"""
        try:
            # Convert PIL image to numpy array
            image_array = np.array(image)
            
            # Perform OCR
            results = self.reader.readtext(image_array)
            
            # Extract text from results
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter by confidence
                    extracted_text.append(text.strip())
            
            # Join all text with spaces
            full_text = " ".join(extracted_text)
            
            # Clean up text
            cleaned_text = self._clean_text(full_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error in text extraction: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Arabic and English
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using Hugging Face models"""
        try:
            if not text or not self.model or not self.tokenizer:
                return text
            
            # Get source and target language codes
            source_code = self.language_codes.get('ar', 'arb_Arab')
            target_code = self.language_codes.get(target_language, 'eng_Latn')
            
            # Prepare input with language codes
            inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate translation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_code],
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode output
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated_text
            
        except Exception as e:
            logger.error(f"Error in translation: {str(e)}")
            # Return original text if translation fails
            return text
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text"""
        if not text:
            return 0.0
        
        # Simple confidence calculation based on text length and content
        # In a real implementation, this would use OCR confidence scores
        
        # Check if text contains Arabic characters
        arabic_chars = len(re.findall(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text))
        total_chars = len(text.replace(' ', ''))
        
        if total_chars == 0:
            return 0.0
        
        # Higher confidence if Arabic characters are present
        arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
        
        # Base confidence
        base_confidence = 0.7
        
        # Adjust based on Arabic character presence
        if arabic_ratio > 0.3:
            confidence = base_confidence + (arabic_ratio * 0.3)
        else:
            confidence = base_confidence * 0.8
        
        return min(confidence, 1.0)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported target languages"""
        return list(self.language_codes.keys())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "ocr_engine": "EasyOCR",
            "translation_model": "NLLB-200" if self.model else "None",
            "gpu_available": torch.cuda.is_available(),
            "supported_languages": self.get_supported_languages()
        }