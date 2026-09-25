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
- **Async infrastructure:** Celery + Redis PDF ingestion worker

## Measured Experiments

### Local Embedding A/B Study

On September 25, 2026, a paired six-run study compared the pinned local Hugging Face
`intfloat/multilingual-e5-base` checkpoint with Ollama `qwen3-embedding:0.6b` at
Top-K 5, 10, and 20.
All variants used the same 20 source-backed project cases and the same fingerprinted
corpus of 113 files / 467 chunks (`9e2d36538c04d6d4...`).
MLflow experiment: `rag-embedding-topk-ablation-2026-09-25`, study ID
`6e138eddd2c745709942a82c0ac3c170`.

| Embedding | Top-K | Hit Rate | Precision | Recall | MRR | nDCG |
|---|---:|---:|---:|---:|---:|---:|
| Hugging Face multilingual-e5-base | 5 | 0.7500 | 0.1500 | 0.7500 | 0.5767 | 0.6190 |
| Hugging Face multilingual-e5-base | 10 | 0.9000 | 0.0900 | 0.9000 | 0.5944 | 0.6651 |
| Hugging Face multilingual-e5-base | 20 | 1.0000 | 0.0500 | 1.0000 | 0.6010 | 0.6900 |
| Ollama qwen3-embedding:0.6b | 5 | 0.3000 | 0.0600 | 0.3000 | 0.1725 | 0.2040 |
| Ollama qwen3-embedding:0.6b | 10 | 0.5500 | 0.0550 | 0.5500 | 0.2069 | 0.2859 |
| Ollama qwen3-embedding:0.6b | 20 | 0.8500 | 0.0425 | 0.8500 | 0.2282 | 0.3624 |

At Top-K 5, multilingual-e5-base improved hit rate and recall by 45 percentage
points (150% relative) over the current Qwen3 embedding on this dataset.
At Top-K 20, it reached 100% hit rate and recall, while precision fell to 5% because
the retriever returned more non-relevant sources.
Increasing Top-K from 5 to 20 for multilingual-e5-base raised recall from 75% to
100% but lowered precision from 15% to 5%.
For this small project-source evaluation, multilingual-e5-base is the stronger
retrieval candidate, while Top-K 10 is a reasonable recall/precision compromise to
validate against the untouched holdout before changing the serving default.
These are retrieval-only measurements; generation quality and latency were not
measured in this study.
See [`docs/evaluation_harness.md`](docs/evaluation_harness.md) for the runnable,
MLflow-tracked command.

### PDF Ingestion with Celery

PDF uploads now return an accepted job ID while Redis queues extraction, embedding,
and Qdrant indexing for the `rag-worker` service.
The real `cs229-probability_review.pdf` (SHA-256
`638ae8e8274ae239a4272a55b296a8c3...`, 91 chunks) was used for three alternating-order
synchronous/Celery pairs, all tracked in MLflow experiment
`rag-pdf-ingestion-ablation-2026-09-25` (parent run
`4525a1ff6e4a47f0b4f101cce5095d4a`).

| Measurement | Synchronous | Celery |
|---|---:|---:|
| Mean end-to-end ingestion | 21.912 s | 21.564 s |
| Median end-to-end ingestion | 21.547 s | 21.539 s |
| Mean queue admission | n/a | 0.028 s |
| Mean worker processing | n/a | 21.318 s |

The median end-to-end difference was only 0.04% lower with Celery, so this small
three-pair run does not establish a processing speedup.
The API accepted the upload in about 28 ms on average and returned while the roughly
21-second workload ran in the worker, which is the measured responsiveness benefit.
The 1.6% lower Celery mean is sensitive to the first synchronous trial and is not
claimed as a robust latency reduction.
See [`docs/evaluation_harness.md`](docs/evaluation_harness.md) for the runnable
benchmark and job-status API.
