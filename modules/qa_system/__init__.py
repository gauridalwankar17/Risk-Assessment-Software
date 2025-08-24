"""
Question-Answering System Module

This module provides:
- Document processing and text extraction
- Multilingual sentence embeddings
- RAG (Retrieval-Augmented Generation) engine
"""

from .document_processor import DocumentProcessor
from .embedding_engine import EmbeddingEngine
from .rag_engine import RAGEngine

__all__ = ["DocumentProcessor", "EmbeddingEngine", "RAGEngine"]