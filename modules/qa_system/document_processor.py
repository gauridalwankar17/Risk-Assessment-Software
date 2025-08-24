"""
Document Processor for PDF and Text Documents

Handles text extraction from various document formats including PDFs, Word documents, and plain text.
"""

import os
from typing import List, Dict, Any
import yaml

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentProcessor:
    """Process various document formats and extract text content."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize document processor."""
        self.config = self._load_config(config_path)
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)['qa_system']
        except FileNotFoundError:
            return {'chunk_size': 1000, 'chunk_overlap': 200}
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not available")
        
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
        
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available")
        
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from any supported document format."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif file_ext in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_document(self, file_path: str) -> List[str]:
        """Process document and return text chunks."""
        text = self.extract_text(file_path)
        return self.chunk_text(text)
    
    def batch_process(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Process multiple documents."""
        results = {}
        for file_path in file_paths:
            try:
                chunks = self.process_document(file_path)
                results[file_path] = chunks
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                results[file_path] = []
        
        return results