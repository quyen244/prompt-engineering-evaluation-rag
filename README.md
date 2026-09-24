# Prompt Engineering and RAG Evaluation

A learning and engineering project for building a provenance-aware retrieval-augmented generation (RAG) system. The repository contains a composable LangChain RAG pipeline, offline retrieval evaluation, prompt optimization experiments, a FastAPI service, a React console, and privacy-conscious observability contracts.

See the [roadmap](ROADMAP.md) for current progress and the next implementation steps.

## Current status

The local Compose API now ingests PDFs with the configured Ollama embedding model, stores vectors in persistent Qdrant, and answers questions with the local Ollama chat model. This path was exercised end to end; other API records and product workflows remain scaffolded. See `ROADMAP.md` for the remaining production work.

## Repository map

```text
src/rag/             Composable LangChain RAG components and pipeline
src/eval/            Source-backed retrieval evaluation harness and metrics
src/rag/optimization/ GEPA prompt optimization integration
src/telemetry/       Privacy-safe events, metrics, logging, and tracing
src/server.py        FastAPI service
src/app.py           Streamlit RAG interface
src/app_cli.py       Command-line RAG interface
frontend/            React and TypeScript lab console
docs/                Setup, evaluation, optimization, serving, and telemetry guides
eval_data/           Evaluation source cases, manifests, and run artifacts
schemas/telemetry/   Versioned telemetry schemas
infra/               Prometheus, Grafana, and Tempo configuration
ROADMAP.md           Project phases, status, and remaining work
```

## Requirements and setup

Python 3.11 is the documented local environment. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements/base.txt
```

On Windows PowerShell, create and activate the environment with:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If using an OpenRouter-backed model, copy `.env.example` to `.env` and set
`OPENROUTER_API_KEY`. Keep secrets out of source control.

## Run locally

Run the Streamlit interface:

```bash
streamlit run src/app.py
```

Run the command-line interface:

```bash
python -m src.app_cli
```

Run the FastAPI service:

```bash
uvicorn src.server:app --reload --port 8000
```

The API health endpoint is `http://localhost:8000/api/health`; its OpenAPI page is
`http://localhost:8000/docs`. Run the React console separately:

```bash
cd frontend
npm ci
npm run dev
```

The frontend reads its backend base URL from `VITE_BACKEND_URL` (see
`frontend/vite.config.ts`).

## Evaluation

The evaluation harness has three source-defined modes: a 20-case checked-in smoke
set, 30 SQuAD improvement cases, and a disjoint 30-case SQuAD validation set. The
Hugging Face rows are fetched at runtime and checked against the pinned manifest;
they are not silently replaced with generated examples. See
[`docs/evaluation_harness.md`](docs/evaluation_harness.md) for provenance and API
usage, and [`eval_data/manifests/evaluation_manifest.json`](eval_data/manifests/evaluation_manifest.json)
for the selected data.

Retrieval evaluation provides deterministic Hit Rate, Precision, Recall, MRR, and
nDCG metrics. The harness does not by itself establish generation quality or a
winning production configuration. Keep measured results tied to saved run artifacts
and held-out validation.

## RAG architecture

The RAG components in `src/rag/` separate ingestion, splitting, embeddings,
retrieval, generation, evaluation, and lineage. The pipeline supports dense and
BM25 retrieval, reciprocal-rank fusion, optional cross-encoder reranking, and
ranked source metadata. See [`docs/rag_core.md`](docs/rag_core.md) for the design
and a minimal example.

Prompt optimization code and the frozen-evidence GEPA adapter live in
`src/rag/optimization/`; implementation notes are in [`docs/gepa.md`](docs/gepa.md).
The lineage and promotion contracts are in `src/rag/lineage.py`.

## Docker services

Docker Compose configuration is provided for the API, frontend, Streamlit,
MLflow, Qdrant, Redis, Prometheus, Grafana, and Tempo:

```bash
docker compose up --build
```

This starts the configured services; it does not imply that every planned workflow
is complete. In particular, verify ingestion persistence, worker behavior, and
end-to-end telemetry before using the stack as a production deployment. The local
service URLs and launcher notes are in `start_docker.sh` and `start_docker.bat`.

## Guides

- [Setup](docs/00_setup.md)
- [Evaluation](docs/01_evaluation.md)
- [Prompt optimization](docs/02_prompt_optimization.md)
- [A/B testing](docs/03_ab_testing.md)
- [Serving](docs/04_serving.md)
- [Observability](docs/05_observability.md)
- [RAG architecture](docs/rag_core.md)
- [Source-backed evaluation harness](docs/evaluation_harness.md)
- [Telemetry contracts](docs/telemetry_contracts.md)

Some older phase guides describe the original learning plan and may mention practice
files or planned modules that are not present in the current repository. The roadmap
tracks current implementation status.

## Technology

- **RAG:** LangChain, Qdrant client, sentence-transformers, OpenRouter-compatible chat models
- **Evaluation and optimization:** Python metrics, MLflow, GEPA
- **Serving and UI:** FastAPI, Streamlit, React, TypeScript
- **Observability:** OpenTelemetry, Prometheus, Grafana, Tempo, structured logging
- **Async infrastructure:** Celery and Redis dependencies/configuration (worker workflow remains planned)
