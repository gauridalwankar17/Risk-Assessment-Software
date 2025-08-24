import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
from typing import Dict, List, Any, Optional
import io
import re
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

logger = logging.getLogger(__name__)

class RAGSystem:
    """
    Retrieval-Augmented Generation system for multilingual Q&A
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """Initialize RAG system with embedding model and FAISS index"""
        try:
            # Initialize sentence transformer for multilingual embeddings
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Embedding model {model_name} loaded successfully")
            
            # Initialize FAISS index
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Document storage
            self.documents = []
            self.document_embeddings = []
            self.document_metadata = []
            
            # Load existing index if available
            self._load_existing_index()
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    def _load_existing_index(self):
        """Load existing FAISS index and documents if available"""
        try:
            index_path = "models/faiss_index.bin"
            docs_path = "models/documents.json"
            
            if os.path.exists(index_path) and os.path.exists(docs_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load documents
                with open(docs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', [])
                    self.document_embeddings = data.get('embeddings', [])
                    self.document_metadata = data.get('metadata', [])
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
    
    def _save_index(self):
        """Save FAISS index and documents"""
        try:
            os.makedirs("models", exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, "models/faiss_index.bin")
            
            # Save documents
            data = {
                'documents': self.documents,
                'embeddings': self.document_embeddings,
                'metadata': self.document_metadata
            }
            
            with open("models/documents.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def add_document(self, document_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Add a PDF document to the RAG system
        
        Args:
            document_content: Raw PDF bytes
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Extract text from PDF
            extracted_text = self._extract_pdf_text(document_content)
            
            if not extracted_text:
                return {
                    "success": False,
                    "error": "No text extracted from PDF",
                    "filename": filename
                }
            
            # Split text into chunks
            text_chunks = self._split_text_into_chunks(extracted_text)
            
            # Generate embeddings for chunks
            chunk_embeddings = self._generate_embeddings(text_chunks)
            
            # Add to FAISS index
            self._add_to_index(chunk_embeddings, text_chunks, filename)
            
            # Save updated index
            self._save_index()
            
            return {
                "success": True,
                "filename": filename,
                "chunks_processed": len(text_chunks),
                "total_documents": len(self.documents),
                "message": f"Document {filename} added successfully"
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "filename": filename
            }
    
    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            text_content = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                if text.strip():
                    text_content.append(text.strip())
            
            pdf_document.close()
            
            return "\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, min(end + 100, len(text))):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks"""
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            return embeddings.astype('float32')
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.array([])
    
    def _add_to_index(self, embeddings: np.ndarray, chunks: List[str], filename: str):
        """Add embeddings and chunks to FAISS index"""
        if len(embeddings) == 0:
            return
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.document_embeddings.append(embeddings[i].tolist())
            self.document_metadata.append({
                'filename': filename,
                'chunk_index': i,
                'chunk_size': len(chunk)
            })
    
    def ask_question(self, question: str, language: str = "en", top_k: int = 3) -> Dict[str, Any]:
        """
        Ask a question and retrieve relevant answers
        
        Args:
            question: User question
            language: Language of the question
            top_k: Number of top results to return
            
        Returns:
            Dictionary with answers and metadata
        """
        try:
            if not self.documents:
                return {
                    "success": False,
                    "error": "No documents indexed. Please upload documents first.",
                    "answers": [],
                    "question": question
                }
            
            # Generate embedding for question
            question_embedding = self.embedding_model.encode([question])
            question_embedding = question_embedding.astype('float32')
            
            # Search FAISS index
            similarities, indices = self.index.search(question_embedding, min(top_k, len(self.documents)))
            
            # Get relevant documents
            answers = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.documents):
                    answer = {
                        "rank": i + 1,
                        "similarity_score": float(similarity),
                        "text": self.documents[idx],
                        "metadata": self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                    }
                    answers.append(answer)
            
            return {
                "success": True,
                "question": question,
                "language": language,
                "answers": answers,
                "total_results": len(answers),
                "index_size": len(self.documents)
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question,
                "answers": []
            }
    
    def search_similar(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for similar documents"""
        return self.ask_question(query, top_k=top_k)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents"""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.documents),
            "index_dimension": self.dimension,
            "index_type": "FAISS FlatIP",
            "unique_files": len(set([meta.get('filename', '') for meta in self.document_metadata if meta]))
        }
    
    def clear_index(self) -> Dict[str, Any]:
        """Clear all indexed documents"""
        try:
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Clear document storage
            self.documents.clear()
            self.document_embeddings.clear()
            self.document_metadata.clear()
            
            # Save empty index
            self._save_index()
            
            return {
                "success": True,
                "message": "Index cleared successfully"
            }
            
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_list(self) -> List[str]:
        """Get list of indexed document filenames"""
        filenames = set()
        for meta in self.document_metadata:
            if meta and 'filename' in meta:
                filenames.add(meta['filename'])
        return list(filenames)