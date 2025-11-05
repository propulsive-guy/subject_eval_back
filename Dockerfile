# Python base
FROM python:3.11-slim

# Prevent Python from writing pyc files / buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OCR / PDF / transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (use Docker caching)
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Cloud Run default port
ENV PORT=8080
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
