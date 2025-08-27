from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Optional, List
import json

from modules.ocr import OCRTranslator
from modules.rag import RAGSystem
from modules.forecast import TimeSeriesForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mini Arabic Industrial Co-Pilot",
    description="AI assistant with OCR translation, RAG Q&A, and time-series forecasting capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
ocr_translator = OCRTranslator()
rag_system = RAGSystem()
forecaster = TimeSeriesForecaster()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Mini Arabic Industrial Co-Pilot API",
        "version": "1.0.0",
        "endpoints": {
            "/translate": "OCR + Translation endpoint",
            "/ask": "RAG Question-Answering endpoint", 
            "/forecast": "Time-Series Forecasting endpoint"
        }
    }

@app.post("/translate")
async def translate_image(
    image: UploadFile = File(...),
    target_language: str = Form(default="en")
):
    """
    Extract Arabic text from image and translate to target language
    """
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image content
        image_content = await image.read()
        
        # Process OCR and translation
        result = ocr_translator.process_image(
            image_content, 
            image.filename,
            target_language
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in translate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    language: str = Form(default="en"),
    top_k: int = Form(default=3)
):
    """
    Ask a question and get answers from indexed documents
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process question through RAG system
        result = rag_system.ask_question(
            question=question,
            language=language,
            top_k=top_k
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/upload_document")
async def upload_document(
    document: UploadFile = File(...)
):
    """
    Upload and index a PDF document for RAG system
    """
    try:
        if not document.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read document content
        document_content = await document.read()
        
        # Process and index document
        result = rag_system.add_document(
            document_content,
            document.filename
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in upload_document endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/forecast")
async def forecast_timeseries(
    csv_file: UploadFile = File(...),
    target_column: str = Form(...),
    forecast_periods: int = Form(default=30),
    detect_anomalies: bool = Form(default=True)
):
    """
    Upload CSV with time-series data and generate forecasts
    """
    try:
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        csv_content = await csv_file.read()
        
        # Process forecasting
        result = forecaster.forecast(
            csv_content,
            csv_file.filename,
            target_column,
            forecast_periods,
            detect_anomalies
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )