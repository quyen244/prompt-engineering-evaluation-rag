# RAG and Prompt Engineering Roadmap

## Goal

Build a usable RAG application where a user can ingest documents and ask questions
with traceable source evidence. Surround that workflow with reproducible evaluation,
prompt and retrieval experiments, reliable serving, privacy-safe observability, and
asynchronous ingestion.

The project should produce defensible measurements. A dashboard fixture or example
value is not a result: report quality, latency, and cost only when they come from a
recorded run on identified data and configuration.

## Where the project is now

The original roadmap described an earlier Streamlit prototype and listed many
practice files that are not in the current repository. The codebase has moved on:
it now includes a modular LangChain RAG foundation, a provenance-checked retrieval
evaluation harness, a GEPA integration, a FastAPI service and React console, and
telemetry contracts and infrastructure configuration. Several app endpoints and
UI records remain scaffolded or in-memory, and those components do not yet form a
complete production document-ingestion workflow.

| Phase | Area | Current status | Evidence / remaining gap |
|---|---|---|---|
| 0 | RAG foundation | **Implemented foundations** | `src/rag/` has composable ingestion, splitting, embedding, retrieval, generation, and pipeline modules. Finish environment and persistent-document integration checks. |
| 1 | Evaluation | **Retrieval harness implemented; generation evaluation partial** | `src/eval/` provides 20 project smoke cases, separate 30-case improvement and validation selections, provenance checks, paired comparison, and retrieval metrics. LLM-as-judge and MLflow-backed evaluation reporting remain to be completed. |
| 2 | Prompt optimization | **GEPA integration implemented; product workflow partial** | `src/rag/optimization/` contains GEPA integration for a LangChain runnable with frozen retrieval evidence; lineage and promotion contracts are in `src/rag/lineage.py`. Establish and publish a repeatable held-out improvement result. |
| 3 | RAG experiments | **Planned / partial foundations** | Dense, BM25, hybrid RRF, and optional reranking components exist. A reproducible sweep runner, persisted comparison set, and measured champion decision are still needed. |
| 4 | Serving | **Local PDF ingest/query path verified** | `src/server.py` embeds PDF chunks with configured Ollama, stores them in persistent Qdrant, retrieves evidence, and generates cited answers. Document history and other product workflows remain partial. |
| 5 | Observability | **Local request telemetry verified** | An actual PDF upload and query emitted stage metrics to Prometheus and query spans to Tempo. Dashboard and alert rules are provisioned; alert firing and production deployment still need separate checks. |
| 6 | Asynchronous ingestion | **Infrastructure prerequisites only** | Celery and Redis dependencies plus a Redis Compose service exist. No end-to-end worker, job status API, retry/idempotency flow, or cross-process trace workflow is implemented yet. |
| 7 | Production integration | **Local Compose flow verified; production work remains** | The local PDF upload-to-query flow runs through Compose. Security limits, asynchronous jobs, operational runbooks, and deployment documentation remain. |

Status describes repository evidence, not a claim that every documented example has
been run. Source files and contract tests are the authoritative implementation.

## Target architecture

```text
Browser / CLI
     │
     ▼
 FastAPI ───────────────► RAG pipeline ─────► answer + ranked source evidence
     │                         │
     │ ingest request          ├──► Qdrant (persistent vectors)
     ▼                         └──► chat model provider
 Redis queue
     │
     ▼
 Celery worker ──► parse PDF → split → embed → store

 Application signals ──► OpenTelemetry / Prometheus ──► Tempo / Grafana
 Evaluation runs ──────► source-backed metrics and experiment history
```

This is the intended system shape. The queue and complete upload-to-query path are
not yet wired end to end.

## Phases and completion criteria

### Phase 0 — Reliable RAG foundation

**Status: foundations implemented; integration verification remains.**

Keep the model-provider boundaries explicit, make pipeline configuration immutable
per instance, and return stable source metadata with every answer. The reusable
pipeline and component boundaries are under `src/rag/`; see [`docs/rag_core.md`](docs/rag_core.md).

Complete this phase when:

- [ ] A clean documented environment installs and starts successfully.
- [ ] Local embeddings work without calling an embedding API.
- [ ] A configured persistent vector store survives an application restart.
- [ ] Each returned source has a stable document identity and useful location metadata.
- [ ] Provider and retrieval configuration can be changed without editing pipeline logic.

### Phase 1 — Reproducible evaluation

**Status: retrieval evaluation implemented; generation judging and experiment reporting remain.**

The current source-backed harness is documented in
[`docs/evaluation_harness.md`](docs/evaluation_harness.md). It defines a checked-in
20-case smoke set and disjoint 30-case SQuAD improvement and validation selections.
It verifies upstream row identity and content hashes and fails on missing data rather
than fabricating fallback cases. Retrieval metrics currently include Hit Rate,
Precision, Recall, MRR, and nDCG.

Next work:

- [ ] Add a generation evaluation path with explicit judge rubric, structured output, and error handling.
- [ ] Record evaluator configuration, dataset identity, metrics, and artifacts in MLflow or an equivalent durable run store.
- [ ] Retain paired case-level outcomes and compare candidates on the same cases.
- [ ] Establish repeatability and judge reliability before publishing quality claims.

### Phase 2 — Prompt optimization and lineage

**Status: GEPA adapter and lineage contracts implemented; measured held-out validation remains.**

The adapter in `src/rag/optimization/` optimizes prompts against frozen retrieval
evidence so candidate prompts are compared on the same context. The lineage module
tracks candidate identity and promotion decisions. See [`docs/gepa.md`](docs/gepa.md).

Complete this phase when:

- [ ] A baseline and optimized prompt are evaluated against the same fixed cases.
- [ ] The candidate is evaluated on untouched validation cases before promotion.
- [ ] Prompt, model, dataset, budget, scorer, and run identities are recorded.
- [ ] Any reported improvement is backed by saved run outputs and a paired comparison.

### Phase 3 — Retrieval and configuration experiments

**Status: retrieval building blocks exist; systematic sweeps and measured selection remain.**

Compare chunking, embedding, dense and BM25 retrieval, hybrid reciprocal-rank fusion,
reranking, and `top_k`. Start with controlled changes, then run a bounded matrix. Use
the improvement split to select candidates and the separate validation split for a
final paired comparison.

Complete this phase when:

- [ ] Experiment parameters and evaluation data are saved for every run.
- [ ] At least one baseline and multiple alternatives have comparable recorded results.
- [ ] Quality, latency, and cost are reported together when measured.
- [ ] A champion is selected with a written rationale and validation evidence.

### Phase 4 — Serving and user workflow

**Status: FastAPI and React scaffolding exists; live pipeline integration remains.**

The API is in `src/server.py`; the frontend is in `frontend/`. Replace scaffolded or
in-memory query, document, experiment, and monitoring records with real service
behavior backed by the RAG pipeline and durable state.

Complete this phase when:

- [ ] The API accepts a question and returns an answer, source evidence, and request identity from the real pipeline.
- [ ] Document upload and query use the same persisted document collection.
- [ ] Health and readiness reflect service dependencies and model/pipeline startup.
- [ ] Errors, timeouts, and request limits have documented behavior.
- [ ] The React console displays API results and clearly distinguishes unavailable data from measured values.

### Phase 5 — Privacy-safe observability

**Status: schemas, instrumentation primitives, and infrastructure configuration exist; end-to-end proof remains.**

The telemetry boundaries and data rules are defined in
[`docs/telemetry_contracts.md`](docs/telemetry_contracts.md). Keep raw questions,
prompts, answers, credentials, and exception stacks out of telemetry sinks. Use
bounded label values and keep identifiers out of Prometheus labels.

Complete this phase when:

- [ ] Real API requests emit correlated request, retrieval, generation, and error signals.
- [ ] Prometheus scrapes the intended metrics; Tempo receives traces; Grafana loads the provisioned dashboard and alerts.
- [ ] Dashboard values derive from live or recorded observations rather than fixtures.
- [ ] Redaction and schema contracts continue to pass whenever telemetry changes.

### Phase 6 — Asynchronous document ingestion

**Status: planned.**

Use Celery with Redis for longer document ingestion jobs. Make tasks idempotent,
report progress, and retry transient failures with bounded backoff. Keep interactive
question answering synchronous unless a measured product need calls for otherwise.

Complete this phase when:

- [ ] Upload returns a job identifier promptly and the API exposes status and progress.
- [ ] A worker parses, splits, embeds, and persists a document.
- [ ] Retries and duplicate submissions do not create inconsistent stored documents.
- [ ] Trace context and safe correlation identifiers cross the worker boundary.

### Phase 7 — Production integration and deployment

**Status: Compose scaffolding exists; production workflow and operational hardening remain.**

Bring the API, frontend, document worker, vector store, tracking, and telemetry
services together. Add file validation, upload size limits, authentication/rate
limits appropriate to the deployment, persistent volumes, service health checks, and
clear configuration instructions.

Complete this phase when:

- [ ] A clean checkout can be configured and started using documented commands.
- [ ] A PDF can be uploaded, indexed persistently, queried, and cited in the UI.
- [ ] Restart and failure behavior are documented and verified.
- [ ] Evaluation and telemetry contracts are part of the project verification process.
- [ ] README results include only reproducible, attributable measurements.

## Recommended next steps

1. Finish a real, persistent upload-to-query path through the current RAG pipeline.
2. Connect the existing FastAPI endpoints to the pipeline and replace in-memory records.
3. Run the 20-case smoke evaluation through the actual retriever; save the run artifact.
4. Add generation evaluation and a repeatable paired prompt/retrieval comparison.
5. Validate live telemetry flow through Prometheus, Tempo, and Grafana.
6. Add Celery ingestion once the synchronous pipeline and API contract are stable.

## Results to publish

Fill this table only from saved runs. Record the data split and configuration alongside
each result; leave a value blank until it has been measured.

| Metric | Baseline | Selected candidate | Validation result |
|---|---:|---:|---:|
| Hit Rate @5 | — | — | — |
| nDCG @5 | — | — | — |
| Faithfulness | — | — | — |
| Answer relevancy | — | — | — |
| p50 / p95 latency | — | — | — |
| Cost per 1,000 queries | — | — | — |
| Index build time | — | — | — |

## Project references

- [Setup guide](docs/00_setup.md)
- [Evaluation guide](docs/01_evaluation.md)
- [Prompt optimization guide](docs/02_prompt_optimization.md)
- [A/B testing guide](docs/03_ab_testing.md)
- [Serving guide](docs/04_serving.md)
- [Observability guide](docs/05_observability.md)
- [RAG architecture](docs/rag_core.md)
- [Evaluation harness](docs/evaluation_harness.md)
- [GEPA integration](docs/gepa.md)
- [Telemetry contracts](docs/telemetry_contracts.md)
