"""
OCR Engine for Arabic Text Extraction

Supports both EasyOCR and Tesseract engines for extracting Arabic text from images.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import yaml
import os

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OCREngine:
    """
    OCR Engine for extracting Arabic text from images.
    
    Supports both EasyOCR and Tesseract engines with configurable settings.
    """
    
    def __init__(self, engine: str = "easyocr", config_path: str = "config/config.yaml"):
        """
        Initialize OCR Engine.
        
        Args:
            engine: OCR engine to use ("easyocr" or "tesseract")
            config_path: Path to configuration file
        """
        self.engine = engine.lower()
        self.config = self._load_config(config_path)
        self.reader = None
        
        if self.engine == "easyocr":
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR not available. Install with: pip install easyocr")
            self._init_easyocr()
        elif self.engine == "tesseract":
            if not TESSERACT_AVAILABLE:
                raise ImportError("Tesseract not available. Install with: pip install pytesseract")
            self._init_tesseract()
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['ocr']
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default settings.")
            return {
                'languages': ['ar', 'en'],
                'confidence_threshold': 0.5,
                'gpu': False
            }
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader."""
        languages = self.config.get('languages', ['ar', 'en'])
        gpu = self.config.get('gpu', False)
        
        self.reader = easyocr.Reader(
            languages,
            gpu=gpu,
            model_storage_directory='./models',
            download_enabled=True
        )
        print(f"EasyOCR initialized with languages: {languages}")
    
    def _init_tesseract(self):
        """Initialize Tesseract configuration."""
        # Set Tesseract language for Arabic
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
        print("Tesseract initialized")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        if image is None:
            raise ValueError("Could not read image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text(self, image_path: str, preprocess: bool = True) -> str:
        """
        Extract text from image.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            Extracted text as string
        """
        if self.engine == "easyocr":
            return self._extract_with_easyocr(image_path, preprocess)
        elif self.engine == "tesseract":
            return self._extract_with_tesseract(image_path, preprocess)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")
    
    def _extract_with_easyocr(self, image_path: str, preprocess: bool) -> str:
        """Extract text using EasyOCR."""
        if preprocess:
            image = self.preprocess_image(image_path)
        else:
            image = image_path
        
        # Extract text with confidence scores
        results = self.reader.readtext(image)
        
        # Filter by confidence threshold
        confidence_threshold = self.config.get('confidence_threshold', 0.5)
        filtered_results = [
            result for result in results 
            if result[2] >= confidence_threshold
        ]
        
        # Extract text
        extracted_text = " ".join([result[1] for result in filtered_results])
        
        return extracted_text
    
    def _extract_with_tesseract(self, image_path: str, preprocess: bool) -> str:
        """Extract text using Tesseract."""
        if preprocess:
            image = self.preprocess_image(image_path)
        else:
            image = image_path
        
        # Configure Tesseract for Arabic
        config = '--oem 3 --psm 6 -l ara+eng'
        
        # Extract text
        text = pytesseract.image_to_string(image, config=config)
        
        return text.strip()
    
    def extract_text_with_confidence(self, image_path: str, preprocess: bool = True) -> List[Tuple[str, float]]:
        """
        Extract text with confidence scores.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            
        Returns:
            List of tuples (text, confidence)
        """
        if self.engine == "easyocr":
            if preprocess:
                image = self.preprocess_image(image_path)
            else:
                image = image_path
            
            results = self.reader.readtext(image)
            confidence_threshold = self.config.get('confidence_threshold', 0.5)
            
            filtered_results = [
                (result[1], result[2]) for result in results 
                if result[2] >= confidence_threshold
            ]
            
            return filtered_results
        else:
            # Tesseract doesn't provide confidence scores in the same way
            text = self.extract_text(image_path, preprocess)
            return [(text, 1.0)]  # Assume high confidence for Tesseract
    
    def batch_extract(self, image_paths: List[str], preprocess: bool = True) -> List[str]:
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image file paths
            preprocess: Whether to preprocess images
            
        Returns:
            List of extracted text strings
        """
        results = []
        for image_path in image_paths:
            try:
                text = self.extract_text(image_path, preprocess)
                results.append(text)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append("")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test OCR engine
    try:
        ocr = OCREngine(engine="easyocr")
        print("OCR Engine initialized successfully!")
        
        # Test with a sample image (if available)
        # text = ocr.extract_text("sample_images/arabic_text.jpg")
        # print(f"Extracted text: {text}")
        
    except Exception as e:
        print(f"Error initializing OCR engine: {e}")