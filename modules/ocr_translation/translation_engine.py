"""
Translation Engine for Arabic to English Translation

Supports M2M100 and NLLB-200 models for high-quality Arabic to English translation.
"""

import torch
from transformers import (
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from typing import List, Optional, Union
import yaml
import os


class TranslationEngine:
    """
    Translation Engine for Arabic to English translation.
    
    Supports both M2M100 and NLLB-200 models with configurable settings.
    """
    
    def __init__(self, model_name: str = "facebook/m2m100_418M", config_path: str = "config/config.yaml"):
        """
        Initialize Translation Engine.
        
        Args:
            model_name: Name of the translation model to use
            config_path: Path to configuration file
        """
        self.model_name = model_name
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        
        self._load_model()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['translation']
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default settings.")
            return {
                'source_lang': 'ar',
                'target_lang': 'en',
                'max_length': 512
            }
    
    def _get_device(self) -> torch.device:
        """Get the best available device (GPU or CPU)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    
    def _load_model(self):
        """Load the translation model and tokenizer."""
        try:
            print(f"Loading model: {self.model_name}")
            
            if "m2m100" in self.model_name.lower():
                self._load_m2m100()
            elif "nllb" in self.model_name.lower():
                self._load_nllb()
            else:
                # Generic AutoModel loading
                self._load_generic_model()
            
            # Move model to device
            self.model.to(self.device)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_m2m100(self):
        """Load M2M100 model and tokenizer."""
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Set source and target languages
        source_lang = self.config.get('source_lang', 'ar')
        target_lang = self.config.get('target_lang', 'en')
        
        self.tokenizer.src_lang = source_lang
        self.tokenizer.tgt_lang = target_lang
    
    def _load_nllb(self):
        """Load NLLB model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # NLLB uses language codes like "ara_Arab" for Arabic
        source_lang = self.config.get('source_lang', 'ara_Arab')
        target_lang = self.config.get('target_lang', 'eng_Latn')
        
        self.source_lang = source_lang
        self.target_lang = target_lang
    
    def _load_generic_model(self):
        """Load generic seq2seq model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def translate_to_english(self, arabic_text: str, max_length: Optional[int] = None) -> str:
        """
        Translate Arabic text to English.
        
        Args:
            arabic_text: Arabic text to translate
            max_length: Maximum length of the output (optional)
            
        Returns:
            Translated English text
        """
        if not arabic_text.strip():
            return ""
        
        try:
            if "m2m100" in self.model_name.lower():
                return self._translate_m2m100(arabic_text, max_length)
            elif "nllb" in self.model_name.lower():
                return self._translate_nllb(arabic_text, max_length)
            else:
                return self._translate_generic(arabic_text, max_length)
                
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation error: {e}"
    
    def _translate_m2m100(self, arabic_text: str, max_length: Optional[int]) -> str:
        """Translate using M2M100 model."""
        # Tokenize input
        inputs = self.tokenizer(arabic_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        max_length = max_length or self.config.get('max_length', 512)
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.get_lang_id(self.tokenizer.tgt_lang),
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode output
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation[0]
    
    def _translate_nllb(self, arabic_text: str, max_length: Optional[int]) -> str:
        """Translate using NLLB model."""
        # Tokenize input with language prefix
        inputs = self.tokenizer(
            f"{self.source_lang} {arabic_text}",
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        max_length = max_length or self.config.get('max_length', 512)
        generated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode output
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation[0]
    
    def _translate_generic(self, arabic_text: str, max_length: Optional[int]) -> str:
        """Translate using generic seq2seq model."""
        # Tokenize input
        inputs = self.tokenizer(arabic_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation
        max_length = max_length or self.config.get('max_length', 512)
        generated_tokens = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
        
        # Decode output
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return translation[0]
    
    def batch_translate(self, arabic_texts: List[str], max_length: Optional[int] = None) -> List[str]:
        """
        Translate multiple Arabic texts to English.
        
        Args:
            arabic_texts: List of Arabic texts to translate
            max_length: Maximum length of the output (optional)
            
        Returns:
            List of translated English texts
        """
        translations = []
        for text in arabic_texts:
            try:
                translation = self.translate_to_english(text, max_length)
                translations.append(translation)
            except Exception as e:
                print(f"Error translating text: {e}")
                translations.append(f"Translation error: {e}")
        
        return translations
    
    def translate_with_confidence(self, arabic_text: str, max_length: Optional[int] = None) -> tuple:
        """
        Translate Arabic text to English with confidence score.
        
        Args:
            arabic_text: Arabic text to translate
            max_length: Maximum length of the output (optional)
            
        Returns:
            Tuple of (translated_text, confidence_score)
        """
        # For now, return translation with a default confidence score
        # In a production system, you might implement more sophisticated confidence scoring
        translation = self.translate_to_english(arabic_text, max_length)
        confidence = 0.9  # Placeholder confidence score
        
        return translation, confidence
    
    def get_supported_languages(self) -> dict:
        """Get information about supported languages."""
        if "m2m100" in self.model_name.lower():
            return {
                "model": "M2M100",
                "supported_languages": ["ar", "en", "fr", "de", "es", "zh", "ja", "ko"],
                "source_lang": self.tokenizer.src_lang,
                "target_lang": self.tokenizer.tgt_lang
            }
        elif "nllb" in self.model_name.lower():
            return {
                "model": "NLLB-200",
                "supported_languages": ["ara_Arab", "eng_Latn", "fra_Latn", "deu_Latn"],
                "source_lang": getattr(self, 'source_lang', 'ara_Arab'),
                "target_lang": getattr(self, 'target_lang', 'eng_Latn')
            }
        else:
            return {
                "model": "Generic",
                "supported_languages": "Unknown",
                "source_lang": "Unknown",
                "target_lang": "Unknown"
            }


# Example usage and testing
if __name__ == "__main__":
    # Test translation engine
    try:
        translator = TranslationEngine(model_name="facebook/m2m100_418M")
        print("Translation Engine initialized successfully!")
        
        # Test translation
        arabic_text = "مرحبا بالعالم"
        english_text = translator.translate_to_english(arabic_text)
        print(f"Arabic: {arabic_text}")
        print(f"English: {english_text}")
        
        # Get supported languages
        lang_info = translator.get_supported_languages()
        print(f"Language info: {lang_info}")
        
    except Exception as e:
        print(f"Error initializing translation engine: {e}")