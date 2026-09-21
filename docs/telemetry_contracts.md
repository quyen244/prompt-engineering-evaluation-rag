# Telemetry contracts

The implementation lives in `src/telemetry/` and is deliberately separate from RAG,
evaluation, and UI code.

- `request_id.py` binds request/run IDs with `contextvars`; IDs are never metric labels.
- `redaction.py` emits canonical JSON with question hashes/lengths and answer lengths;
  raw content and credentials are rejected from the event boundary.
- `events.py` defines versioned request and document-experiment records. Every record
  carries retrieval strategy (`bm25_only`, `bm25_reranker`, `dense`, or `hybrid`),
  prompt/model versions, dataset split, run identity, latency, token/cost fields,
  and an explicit outcome. Quality scores are intentionally not invented.
- `experiment_history.py` appends document events to JSONL and counts sink failures.
- `metrics.py` defines each Prometheus metric once and allow-lists label vocabularies.
  IDs and free text belong in traces/events, not time-series labels.
- `tracing.py` uses a batch OTLP processor and fails open; dropped signals are counted.
- `logging.py` produces JSON correlated with the current OTel trace where available.

Prometheus, Tempo, Grafana dashboard, and alerts are under `infra/`. Schema contracts
are under `schemas/telemetry/`. The dashboard uses rates for spend (USD/hour), not a
fabricated cumulative result.
