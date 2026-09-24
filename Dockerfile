FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt /app/requirements.txt

# Install CPU-only torch first to prevent large CUDA download, then project dependencies
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r /app/requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic pydantic-settings python-multipart pypdf prometheus-client opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc structlog streamlit mlflow qdrant-client rank-bm25 tenacity httpx

COPY . /app/

EXPOSE 8000 8501 5000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]
