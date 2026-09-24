# Telemetry contracts

The implementation lives in `src/telemetry/` and is deliberately separate from RAG,
evaluation, and UI code.

- `request_id.py` binds request/run IDs with `contextvars`; IDs are never metric labels.
- `redaction.py` emits canonical JSON with question hashes/lengths and answer lengths;
  raw content and credentials are rejected from the event boundary.
- `events.py` defines versioned request, retrieval, generation, evaluation, error,
  and document-experiment records. Every lifecycle record carries retrieval strategy
  (`bm25_only`, `bm25_reranker`, `dense`, or `hybrid`), prompt/model versions,
  dataset split, run identity, and an explicit outcome. Retrieval records carry only
  count/top-score/context-token measurements; generation records carry only token,
  cost, and finish-reason measurements; error records carry type/code rather than
  exception text or stacks.
- `EvaluationEvent` and `record_evaluation()` are the offline MLflow/evaluation
  hand-off. They accept a supplied score and configured acceptance threshold for a
  bounded metric vocabulary, and emit nothing when a scorer did not produce a score.
  This contract deliberately does not define default scores or thresholds.
- `experiment_history.py` appends document events to JSONL and counts sink failures.
- `metrics.py` defines each Prometheus metric once and allow-lists label vocabularies.
  Uploads can publish bounded outcomes and chunk counts; upload/parse/chunk/embed/index
  durations and failures share the finite stage vocabulary. Never add filenames,
  document IDs, prompts, questions, answers, or exception text as labels.
  IDs and free text belong in traces/events, not time-series labels.
- `tracing.py` uses a batch OTLP processor and fails open; dropped signals are counted.
- `logging.py` produces JSON correlated with the current OTel trace where available.

Prometheus, Tempo, Grafana dashboard, and alerts are under `infra/`. Prometheus scrapes
the API at `/metrics`; the API exports traces to Tempo via OTLP gRPC. Online monitoring
uses ingestion outcomes, request/stage/cost/error signals; offline monitoring publishes actual evaluation
outputs alongside their configured threshold. Schema contracts are under
`schemas/telemetry/`, including `lifecycle-event.schema.json`. The dashboard uses
rates for spend (USD/hour), not a fabricated cumulative result.
