# 🚀 Mini Arabic Industrial Co-Pilot

A comprehensive Python project for Arabic text processing, document Q&A, and time-series forecasting, designed to run seamlessly in Google Colab.

## 🏗️ Project Structure

```
mini_arabic_co_pilot/
├── requirements.txt
├── config/
│   └── config.yaml
├── modules/
│   ├── __init__.py
│   ├── ocr_translation/
│   │   ├── __init__.py
│   │   ├── ocr_engine.py
│   │   └── translation_engine.py
│   ├── qa_system/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── embedding_engine.py
│   │   └── rag_engine.py
│   └── forecasting/
│       ├── __init__.py
│       ├── prophet_forecaster.py
│       └── anomaly_detector.py
├── data/
│   ├── sample_images/
│   ├── sample_documents/
│   └── sample_sensor_data/
├── notebooks/
│   ├── 01_ocr_translation_demo.ipynb
│   ├── 02_qa_system_demo.ipynb
│   └── 03_forecasting_demo.ipynb
├── backend/
│   ├── main.py
│   └── api/
│       ├── __init__.py
│       ├── translate.py
│       ├── ask.py
│       └── forecast.py
├── frontend/
│   └── streamlit_app.py
└── utils/
    ├── __init__.py
    └── helpers.py
```

## 🚀 Quick Start (Google Colab)

### 1. Install Dependencies
```python
!pip install -r requirements.txt
```

### 2. Run OCR + Translation
```python
from modules.ocr_translation.ocr_engine import OCREngine
from modules.ocr_translation.translation_engine import TranslationEngine

# Initialize engines
ocr = OCREngine()
translator = TranslationEngine()

# Process Arabic image
arabic_text = ocr.extract_text("path/to/arabic_image.jpg")
english_text = translator.translate_to_english(arabic_text)
print(f"Translated: {english_text}")
```

### 3. Run Q&A System
```python
from modules.qa_system.rag_engine import RAGEngine

# Initialize RAG engine
rag = RAGEngine()

# Add documents
rag.add_documents(["path/to/manual1.pdf", "path/to/manual2.pdf"])

# Ask questions
answer = rag.ask_question("How do I calibrate the sensor?")
print(f"Answer: {answer}")
```

### 4. Run Forecasting
```python
from modules.forecasting.prophet_forecaster import ProphetForecaster

# Initialize forecaster
forecaster = ProphetForecaster()

# Load sensor data
forecaster.load_data("path/to/sensor_data.csv")

# Generate forecast
forecast = forecaster.forecast(periods=30)
print(forecast.head())
```

## 🛠️ Features

### 📸 OCR + Translation Module
- **Input**: Arabic text images
- **Output**: English text
- **OCR Engines**: EasyOCR, Tesseract
- **Translation**: M2M100, NLLB-200

### ❓ Question-Answering Module
- **Input**: Arabic/English questions + PDFs/manuals
- **Output**: Extracted answers from documents
- **Vector Database**: FAISS
- **Embeddings**: Multilingual sentence embeddings

### 📊 Time-Series Forecasting Module
- **Input**: Sensor CSV files
- **Output**: Forecasts + anomaly detection
- **Algorithm**: Prophet
- **Features**: Trend analysis, seasonality, anomaly detection

## 🔧 Backend (Optional)
- FastAPI endpoints for each module
- RESTful API design
- Easy integration with other systems

## 🎨 Frontend (Optional)
- Streamlit interface with 3 tabs
- Interactive demos for each module
- File upload capabilities

## 📊 MLflow Integration (Optional)
- Experiment tracking
- Model versioning
- Performance metrics

## 🎯 Use Cases

1. **Industrial Documentation**: Translate Arabic manuals to English
2. **Technical Support**: Q&A from technical documents
3. **Predictive Maintenance**: Forecast equipment failures
4. **Quality Control**: Anomaly detection in sensor data

## 🚨 Requirements

- Python 3.8+
- Google Colab environment
- GPU support (optional, for faster processing)

## 📝 License

MIT License - feel free to use and modify!
