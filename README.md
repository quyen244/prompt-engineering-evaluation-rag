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

The current evaluation input is the lightweight TechQA set in [`eval_data/`](eval_data/): 20 questions with reference answers and gold document IDs, plus 100 real documents including 80 distractors.
The retrieval ablation compares BM25, dense, and hybrid retrieval with local Qwen3 and cached multilingual E5 embeddings at several Top-K values.
It records Hit Rate, Precision, Recall, MRR, and nDCG in MLflow, then writes a fingerprinted champion configuration for the generation evaluator.
Generation uses local `qwen3.5:4b` and a separate OpenRouter judge for faithfulness, answer correctness, answer relevancy, and citation precision.
See [`features/techqa-evaluation/spec.md`](features/techqa-evaluation/spec.md) for the study contract and [`docs/evaluation_harness.md`](docs/evaluation_harness.md) for the historical harness whose input files have been retired.

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
- [TechQA evaluation commands](docs/techqa_evaluation.md)
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

### TechQA Retrieval Ablation

On September 26, 2026, 15 configurations were evaluated on the lightweight TechQA corpus of 100 documents and 20 labeled questions.
The shared dataset fingerprint is `458ab06a6266426d1fdc0eb6da50984b694a30bce9000835a695a71d844437ca`.
MLflow experiment `techqa-retrieval`, study ID `4eff253f672d49c98ef4c1328836d94d` contains all five retrieval metrics and per-question document and passage IDs.

| Method | Embedding | Top-K | Hit Rate | MRR | nDCG |
|---|---|---:|---:|---:|---:|
| BM25 | None | 5 | 1.000 | 0.898 | 0.922 |
| Dense | Qwen3 0.6B | 3 | 1.000 | 1.000 | 1.000 |
| Dense | multilingual E5 base | 3 | 0.950 | 0.950 | 0.950 |
| Hybrid | Qwen3 0.6B | 3 | 1.000 | 0.975 | 0.981 |

Dense Qwen ranked the gold document first for all 20 supplied questions at Top-K 3, 5, and 10.
An MLflow tracked K1 confirmation run (`04daa09fe1644e469a98c56e246261ec`) also scored 1.000 on all five retrieval metrics.
The current generation champion is therefore dense Qwen at Top-K 1, recorded in `eval_data/results/techqa_retrieval_champion.json`.
These questions were also used to choose the configuration, and some contain document titles or exact identifiers, so this result does not establish performance on unseen questions.
The full matrix and command are described in [`docs/techqa_evaluation.md`](docs/techqa_evaluation.md).

### Historical Local Embedding A/B Study

On September 25, 2026, a paired six-run study on the former project-source dataset
compared the pinned local Hugging Face
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
The former input files were removed when TechQA became the current evaluation set.
The saved MLflow runs remain historical evidence for that study.

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

### Concurrent Query Load Test

[`scripts/locust_rag.py`](scripts/locust_rag.py) models concurrent Ask requests against already-indexed documents.
Each request measures question embedding, Qdrant retrieval, and local Ollama answer generation separately; uploads are not part of this load profile.
Runs record throughput, failures, overall latency, and per-stage latency in MLflow experiment `rag-query-concurrency`.
The profile stops after 120 seconds or 20 requests by default.
Start with the Locust UI as described in [`docs/load_testing.md`](docs/load_testing.md).
