import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Mini Arabic Industrial Co-Pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 Mini Arabic Industrial Co-Pilot")
    st.markdown("AI-powered assistant for OCR translation, document Q&A, and time-series forecasting")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API status check
        if st.button("Check API Status"):
            check_api_status()
        
        # Model information
        st.subheader("📊 System Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ Cannot connect to API")
        
        st.info("Make sure the FastAPI backend is running on port 8000")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📷 OCR + Translation", 
        "📚 Document Q&A", 
        "📈 Time-Series Forecast"
    ])
    
    with tab1:
        ocr_translation_tab()
    
    with tab2:
        document_qa_tab()
    
    with tab3:
        timeseries_forecast_tab()

def ocr_translation_tab():
    """OCR and Translation tab"""
    st.header("📷 OCR + Translation")
    st.markdown("Upload images containing Arabic text and translate to your preferred language")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing Arabic text"
    )
    
    # Language selection
    col1, col2 = st.columns(2)
    with col1:
        target_language = st.selectbox(
            "Target Language",
            ["en", "ar", "fr", "de", "es", "zh"],
            format_func=lambda x: {
                "en": "English",
                "ar": "Arabic", 
                "fr": "French",
                "de": "German",
                "es": "Spanish",
                "zh": "Chinese"
            }[x]
        )
    
    with col2:
        if st.button("🚀 Process Image", type="primary"):
            if uploaded_file is not None:
                process_ocr_image(uploaded_file, target_language)
            else:
                st.warning("Please upload an image first")

def document_qa_tab():
    """Document Q&A tab"""
    st.header("📚 Document Q&A")
    st.markdown("Upload PDF documents and ask questions in Arabic or English")
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to index for Q&A"
    )
    
    if uploaded_pdf is not None:
        if st.button("📥 Index Document", type="primary"):
            index_document(uploaded_pdf)
    
    # Question asking
    st.subheader("❓ Ask Questions")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area(
            "Enter your question",
            placeholder="Ask a question about your uploaded documents...",
            height=100
        )
    
    with col2:
        language = st.selectbox(
            "Language",
            ["en", "ar"],
            format_func=lambda x: "English" if x == "en" else "العربية"
        )
        
        top_k = st.number_input("Top Results", min_value=1, max_value=10, value=3)
        
        if st.button("🔍 Search", type="primary"):
            if question.strip():
                ask_question(question, language, top_k)
            else:
                st.warning("Please enter a question")

def timeseries_forecast_tab():
    """Time-series forecasting tab"""
    st.header("📈 Time-Series Forecasting")
    st.markdown("Upload CSV data and generate forecasts with anomaly detection")
    
    # File upload
    uploaded_csv = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with time-series data"
    )
    
    if uploaded_csv is not None:
        # Preview data
        try:
            df = pd.read_csv(uploaded_csv)
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column selection
            col1, col2, col3 = st.columns(3)
            
            with col1:
                target_column = st.selectbox(
                    "Target Column",
                    df.columns.tolist(),
                    help="Select the column to forecast"
                )
            
            with col2:
                forecast_periods = st.number_input(
                    "Forecast Periods",
                    min_value=1,
                    max_value=365,
                    value=30,
                    help="Number of periods to forecast"
                )
            
            with col3:
                detect_anomalies = st.checkbox(
                    "Detect Anomalies",
                    value=True,
                    help="Enable anomaly detection"
                )
            
            if st.button("🔮 Generate Forecast", type="primary"):
                generate_forecast(uploaded_csv, target_column, forecast_periods, detect_anomalies)
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

def process_ocr_image(image_file, target_language):
    """Process OCR image through API"""
    try:
        with st.spinner("Processing image..."):
            files = {"image": image_file}
            data = {"target_language": target_language}
            
            response = requests.post(f"{API_BASE_URL}/translate", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    display_ocr_results(result)
                else:
                    st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code}")
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def display_ocr_results(result):
    """Display OCR translation results"""
    st.success("✅ Image processed successfully!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Extracted Text")
        st.text_area(
            "Arabic Text",
            value=result.get("extracted_text", ""),
            height=150,
            disabled=True
        )
        
        st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
        st.metric("Word Count", result.get("word_count", 0))
    
    with col2:
        st.subheader("🌐 Translation")
        st.text_area(
            "Translated Text",
            value=result.get("translated_text", ""),
            height=150,
            disabled=True
        )
        
        st.info(f"Source: Arabic → Target: {result.get('target_language', 'en').upper()}")

def index_document(pdf_file):
    """Index PDF document through API"""
    try:
        with st.spinner("Indexing document..."):
            files = {"document": pdf_file}
            
            response = requests.post(f"{API_BASE_URL}/upload_document", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    st.success(f"✅ Document indexed successfully!")
                    st.info(f"Processed {result.get('chunks_processed', 0)} text chunks")
                else:
                    st.error(f"Indexing failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code}")
                
    except Exception as e:
        st.error(f"Error indexing document: {str(e)}")

def ask_question(question, language, top_k):
    """Ask question through API"""
    try:
        with st.spinner("Searching for answers..."):
            data = {
                "question": question,
                "language": language,
                "top_k": top_k
            }
            
            response = requests.post(f"{API_BASE_URL}/ask", data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    display_qa_results(result)
                else:
                    st.error(f"Search failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code}")
                
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")

def display_qa_results(result):
    """Display Q&A results"""
    st.success(f"✅ Found {result.get('total_results', 0)} relevant answers")
    
    answers = result.get("answers", [])
    
    for i, answer in enumerate(answers):
        with st.expander(f"Answer {answer.get('rank', i+1)} (Score: {answer.get('similarity_score', 0):.3f})"):
            st.markdown(f"**Text:** {answer.get('text', '')}")
            
            metadata = answer.get("metadata", {})
            if metadata:
                st.caption(f"Source: {metadata.get('filename', 'Unknown')} | Chunk: {metadata.get('chunk_index', 'N/A')}")

def generate_forecast(csv_file, target_column, forecast_periods, detect_anomalies):
    """Generate forecast through API"""
    try:
        with st.spinner("Generating forecast..."):
            files = {"csv_file": csv_file}
            data = {
                "target_column": target_column,
                "forecast_periods": forecast_periods,
                "detect_anomalies": detect_anomalies
            }
            
            response = requests.post(f"{API_BASE_URL}/forecast", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    display_forecast_results(result)
                else:
                    st.error(f"Forecast failed: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code}")
                
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

def display_forecast_results(result):
    """Display forecasting results"""
    st.success("✅ Forecast generated successfully!")
    
    # Data summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Points", result.get("data_points", 0))
    
    with col2:
        st.metric("Forecast Periods", result.get("forecast_periods", 0))
    
    with col3:
        anomaly_count = len(result.get("anomalies", []))
        st.metric("Anomalies Detected", anomaly_count)
    
    # Data summary
    data_summary = result.get("data_summary", {})
    if data_summary and "statistics" in data_summary:
        st.subheader("📊 Data Statistics")
        
        stats = data_summary["statistics"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{stats.get('mean', 0):.2f}")
        with col2:
            st.metric("Std Dev", f"{stats.get('std', 0):.2f}")
        with col3:
            st.metric("Min", f"{stats.get('min', 0):.2f}")
        with col4:
            st.metric("Max", f"{stats.get('max', 0):.2f}")
    
    # Anomalies
    anomalies = result.get("anomalies", [])
    if anomalies:
        st.subheader("🚨 Detected Anomalies")
        
        anomaly_df = pd.DataFrame(anomalies)
        st.dataframe(anomaly_df, use_container_width=True)
    
    # Forecast data
    forecast_data = result.get("forecast", [])
    if forecast_data:
        st.subheader("🔮 Forecast Results")
        
        # Convert to DataFrame for display
        forecast_df = pd.DataFrame([
            {
                "Period": item.get("period", 0),
                "Ensemble Forecast": item.get("ensemble", {}).get("yhat", "N/A"),
                "Prophet": item.get("prophet", {}).get("yhat", "N/A"),
                "ARIMA": item.get("arima", {}).get("yhat", "N/A")
            }
            for item in forecast_data
        ])
        
        st.dataframe(forecast_df, use_container_width=True)

def check_api_status():
    """Check API connection status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is running and healthy!")
        else:
            st.error(f"❌ API returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running.")
    except Exception as e:
        st.error(f"❌ Error checking API status: {str(e)}")

if __name__ == "__main__":
    main()