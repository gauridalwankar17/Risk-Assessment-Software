#!/usr/bin/env python3
"""
Google Colab Setup for Mini Arabic Industrial Co-Pilot
Run this in a Colab notebook to set up and run the application
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_colab():
    """Setup the project in Google Colab"""
    print("🚀 Setting up Mini Arabic Industrial Co-Pilot in Google Colab...")
    
    # Install system dependencies
    print("📦 Installing system dependencies...")
    os.system("apt-get update")
    os.system("apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1")
    
    # Install Python dependencies
    print("🐍 Installing Python dependencies...")
    dependencies = [
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "streamlit==1.28.1",
        "easyocr==1.7.0",
        "Pillow==10.1.0",
        "torch==2.1.0",
        "transformers==4.35.2",
        "sentence-transformers==2.2.2",
        "scikit-learn==1.3.2",
        "PyMuPDF==1.23.8",
        "faiss-cpu==1.7.4",
        "prophet==1.1.4",
        "statsmodels==0.14.0",
        "pandas==2.1.3",
        "numpy==1.25.2",
        "plotly==5.17.0",
        "requests==2.31.0"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        os.system(f"pip install {dep}")
    
    print("✅ Dependencies installed successfully!")

def create_colab_notebook():
    """Create a Colab notebook with the setup"""
    notebook_content = '''# 🤖 Mini Arabic Industrial Co-Pilot - Google Colab Setup

## 🚀 Quick Setup

### 1. Install Dependencies
```python
# Run this cell first
!pip install fastapi uvicorn python-multipart streamlit easyocr torch transformers sentence-transformers scikit-learn PyMuPDF faiss-cpu prophet statsmodels pandas numpy plotly requests
!apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

### 2. Download Project Files
```python
# Download the project files
!wget -O main.py https://raw.githubusercontent.com/your-repo/main.py
!wget -O requirements.txt https://raw.githubusercontent.com/your-repo/requirements.txt
!mkdir -p modules frontend
!wget -O modules/ocr.py https://raw.githubusercontent.com/your-repo/modules/ocr.py
!wget -O modules/rag.py https://raw.githubusercontent.com/your-repo/modules/rag.py
!wget -O modules/forecast.py https://raw.githubusercontent.com/your-repo/modules/forecast.py
!wget -O frontend/streamlit_app.py https://raw.githubusercontent.com/your-repo/frontend/streamlit_app.py
```

### 3. Start Backend
```python
# Start FastAPI backend
!python main.py &
```

### 4. Start Frontend
```python
# Start Streamlit frontend
!streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
```

### 5. Access the Application
- Backend API: http://localhost:8000
- Frontend UI: http://localhost:8501

## 📱 Using ngrok for External Access

```python
# Install ngrok
!pip install pyngrok

# Create public URLs
from pyngrok import ngrok
http_tunnel = ngrok.connect(8000)
streamlit_tunnel = ngrok.connect(8501)

print(f"🌐 Backend API: {http_tunnel.public_url}")
print(f"🎨 Frontend UI: {streamlit_tunnel.public_url}")
```

## 🔧 Alternative: Run Everything in One Cell

```python
# Complete setup and run
import os
import subprocess
import time

# Install dependencies
print("Installing dependencies...")
os.system("pip install fastapi uvicorn python-multipart streamlit easyocr torch transformers sentence-transformers scikit-learn PyMuPDF faiss-cpu prophet statsmodels pandas numpy plotly requests")
os.system("apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1")

# Download files (you'll need to upload these manually or use your own URLs)
print("Setting up project...")

# Start services
print("Starting services...")
os.system("python main.py &")
time.sleep(5)
os.system("streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &")

print("✅ Services started!")
print("🌐 Backend: http://localhost:8000")
print("🎨 Frontend: http://localhost:8501")
```

## 📁 Manual File Upload

If the wget commands don't work, manually upload these files to Colab:
1. `main.py`
2. `modules/ocr.py`
3. `modules/rag.py`
4. `modules/forecast.py`
5. `frontend/streamlit_app.py`
6. `requirements.txt`

## ⚠️ Important Notes

- **GPU Runtime**: Enable GPU runtime in Colab for faster inference
- **Session Time**: Colab sessions have time limits (12 hours for free)
- **Storage**: Colab provides ~100GB of storage
- **Models**: Will download automatically on first use

## 🆘 Troubleshooting

- If services don't start, check the output for errors
- Restart runtime if you encounter issues
- Use `!ps aux` to check if processes are running
- Use `!kill -9 <PID>` to stop processes if needed
'''
    
    with open('colab_setup.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("📓 Colab notebook created: colab_setup.ipynb")

if __name__ == "__main__":
    setup_colab()
    create_colab_notebook()