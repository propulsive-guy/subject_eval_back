# Python base
FROM python:3.11-slim

# Prevent Python from writing pyc files / buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OCR / PDF / Transformers / NLP
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ✅ Pre-download NLTK data (fix punkt_tab not found)
RUN python3 - <<EOF
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
EOF

# Copy project files
COPY . .

# Default Cloud Run port
ENV PORT=8080
EXPOSE 8080

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
