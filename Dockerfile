FROM python:3.12-slim

WORKDIR /app

# System deps (minimal; useful for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY rag_server.py /app/rag_server.py
COPY rag_build_index.py /app/rag_build_index.py
COPY rag_extract_chunks.py /app/rag_extract_chunks.py

# Default env (override in compose)
ENV INDEX_DIR=/app/rag_index \
    COLLECTION_NAME=otel_docs_chunks \
    OLLAMA_URL=http://ollama:11434/api/generate \
    OLLAMA_MODEL=llama3.2:1b

EXPOSE 8000

CMD ["uvicorn", "rag_server:app", "--host", "0.0.0.0", "--port", "8000"]

