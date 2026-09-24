#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# RAG System & Observability Stack — 1-Click Docker Compose Launcher
# ==============================================================================
echo "[INFO] Starting all RAG services via Docker Compose..."
docker compose up -d

echo ""
echo "=============================================================================="
echo "[SUCCESS] RAG Production Fleet is Running!"
echo "=============================================================================="
echo "React Console (Vite):         http://localhost:5173/"
echo "FastAPI Serving Gateway:      http://localhost:8000/api/health"
echo "Prometheus Scrape Target:     http://localhost:8000/metrics"
echo "Streamlit Interactive Q&A:    http://localhost:8501/"
echo "MLflow Tracking UI:           http://localhost:5000/"
echo "Prometheus Alerts & Metrics:  http://localhost:9090/"
echo "Grafana Dashboards:           http://localhost:3000/ (admin/admin)"
echo "Grafana Tempo Traces:         http://localhost:3200/"
echo "Qdrant Vector DB:             http://localhost:6333/dashboard"
echo "Redis Task Broker:            redis://localhost:6379/"
echo "=============================================================================="
echo "Architecture Visualizer:      .lavish/system_architecture.html"
echo ""
