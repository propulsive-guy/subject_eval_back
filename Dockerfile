FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download only required NLTK data
RUN python3 -m nltk.downloader punkt stopwords

COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host=0.0.0.0", "--port=8080"]
