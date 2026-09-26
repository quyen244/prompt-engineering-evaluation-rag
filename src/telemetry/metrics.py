"""Single definition point for bounded Prometheus metrics.

Never add request IDs, document IDs, questions, user IDs, or error messages as labels.
Those belong to spans and append-only events.
"""
from __future__ import annotations

import os
from typing import Any

from .constants import PROMETHEUS_LABELS, QUALITY_METRICS, RETRIEVAL_STRATEGIES

try:  # Optional import keeps local/offline tools usable without Prometheus installed.
    from prometheus_client import Counter, Gauge, Histogram
except ImportError:  # pragma: no cover - exercised only in minimal environments
    class _NoopMetric:
        def __init__(self, *_: Any, **__: Any) -> None: pass
        def labels(self, **_: Any) -> "_NoopMetric": return self
        def inc(self, *_: Any, **__: Any) -> None: pass
        def observe(self, *_: Any, **__: Any) -> None: pass
        def set(self, *_: Any, **__: Any) -> None: pass
    Counter = Gauge = Histogram = _NoopMetric  # type: ignore[misc,assignment]

RAG_DURATION = Histogram(
    "rag_query_duration_seconds", "RAG latency by stage", ["stage", "retrieval_strategy"],
    # histogram_quantile cannot extrapolate past the largest finite bucket, so a
    # ceiling below real generation latency silently reports that ceiling as the
    # quantile. Measured generation p95 under 20 concurrent users was 163s.
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 120, 240, 480, float("inf")),
)
RAG_TOP_SCORE = Histogram(
    "rag_retrieval_top_score", "Top retrieval score", ["retrieval_strategy"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
RAG_ERRORS = Counter("rag_errors_total", "RAG errors", ["stage", "error_type"])
RAG_INGESTIONS = Counter("rag_ingestions_total", "Document ingestion attempts", ["outcome"])
RAG_INGESTED_CHUNKS = Histogram(
    "rag_ingested_chunks", "Number of chunks produced by an ingestion",
    buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, float("inf")),
)
RAG_NO_ANSWER = Counter("rag_no_answer_total", "Successful requests with no answer")
RAG_IN_FLIGHT = Gauge("rag_queries_in_flight", "RAG requests currently running")
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["token_type", "model"])
LLM_COST = Counter("llm_cost_usd_total", "LLM spend in USD", ["model", "endpoint"])
CONTEXT_TOKENS = Histogram(
    "rag_context_tokens", "Retrieved context token count",
    buckets=(0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, float("inf")),
)
TELEMETRY_DROPPED = Counter("telemetry_dropped_signals_total", "Signals dropped by telemetry", ["signal"])
RAG_EVALUATION_SCORE = Gauge(
    "rag_evaluation_score", "Offline evaluation score supplied by the evaluator",
    ["quality_metric", "dataset_split", "retrieval_strategy"],
)
RAG_EVALUATION_THRESHOLD = Gauge(
    "rag_evaluation_threshold", "Configured offline evaluation acceptance threshold",
    ["quality_metric", "dataset_split"],
)

_ALLOWED_MODELS = frozenset(filter(None, os.getenv("TELEMETRY_MODELS", "default,other").split(",")))
_ALLOWED_ENDPOINTS = frozenset({"query", "document_experiment"})


def retrieval_strategy(value: str) -> str:
    aliases = {
        "bm25-only": "bm25_only", "bm25": "bm25_only", "bm25_only": "bm25_only",
        "bm25-plus-reranker": "bm25_reranker", "bm25_reranker": "bm25_reranker",
        "dense": "dense", "hybrid": "hybrid",
    }
    try:
        result = aliases[value.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported retrieval strategy: {value!r}") from exc
    if result not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"unsupported retrieval strategy: {value!r}")
    return result


def _label(name: str, value: str) -> str:
    allowed = PROMETHEUS_LABELS.get(name)
    if allowed is not None and value not in allowed:
        raise ValueError(f"unbounded or unsupported Prometheus label {name}={value!r}")
    return value


def model_label(model: str) -> str:
    # Model names are deliberately allow-listed; arbitrary model/user input must not
    # create unbounded time series.
    return model if model in _ALLOWED_MODELS else "other"


def record_stage_duration(stage: str, seconds: float, strategy: str) -> None:
    RAG_DURATION.labels(stage=_label("stage", stage), retrieval_strategy=retrieval_strategy(strategy)).observe(max(0.0, seconds))


def record_ingestion(outcome: str, *, chunk_count: int | None = None) -> None:
    """Record one upload/index outcome and optional bounded chunk count."""
    if outcome not in {"success", "error", "timeout", "cancelled"}:
        raise ValueError(f"unsupported ingestion outcome: {outcome!r}")
    if chunk_count is not None:
        if isinstance(chunk_count, bool) or not isinstance(chunk_count, int) or chunk_count < 0:
            raise ValueError("chunk_count must be a non-negative integer")
        if outcome != "success":
            raise ValueError("chunk_count is only recorded for successful ingestion")
        RAG_INGESTED_CHUNKS.observe(chunk_count)
    RAG_INGESTIONS.labels(outcome=outcome).inc()


def record_error(stage: str, error_type: str) -> None:
    """Record an error by finite lifecycle stage and error category only."""
    RAG_ERRORS.labels(
        stage=_label("stage", stage), error_type=_label("error_type", error_type),
    ).inc()


def record_llm_call(model: str, prompt_tokens: int, completion_tokens: int, *, endpoint: str = "query") -> float:
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if endpoint not in _ALLOWED_ENDPOINTS:
        raise ValueError(f"unsupported endpoint label: {endpoint}")
    safe_model = model_label(model)
    LLM_TOKENS.labels(token_type="prompt", model=safe_model).inc(prompt_tokens)
    LLM_TOKENS.labels(token_type="completion", model=safe_model).inc(completion_tokens)
    # Pricing must be supplied by the caller/configuration; no fabricated quality or
    # cost is emitted when pricing is unavailable.
    return float(prompt_tokens + completion_tokens)


def record_cost(model: str, cost_usd: float, *, endpoint: str = "query") -> None:
    if cost_usd < 0:
        raise ValueError("cost_usd must be non-negative")
    if endpoint not in _ALLOWED_ENDPOINTS:
        raise ValueError(f"unsupported endpoint label: {endpoint}")
    LLM_COST.labels(model=model_label(model), endpoint=endpoint).inc(cost_usd)


def mark_dropped(signal: str) -> None:
    safe_signal = signal if signal in {"span", "event", "log"} else "other"
    TELEMETRY_DROPPED.labels(signal=safe_signal).inc()


def record_evaluation(
    quality_metric: str, score: float | None, threshold: float | None, *,
    dataset_split: str, strategy: str,
) -> None:
    """Publish an actual offline score and its configured threshold when supplied.

    ``None`` deliberately emits nothing, so missing MLflow/judge output cannot be
    mistaken for a passing quality result.
    """
    if quality_metric not in QUALITY_METRICS:
        raise ValueError(f"unsupported quality metric: {quality_metric}")
    if dataset_split not in PROMETHEUS_LABELS["dataset_split"]:
        raise ValueError(f"unsupported dataset split: {dataset_split}")
    for name, value in (("score", score), ("threshold", threshold)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1):
            raise ValueError(f"evaluation {name} must be between 0 and 1")
    labels = {"quality_metric": quality_metric, "dataset_split": dataset_split}
    if score is not None:
        RAG_EVALUATION_SCORE.labels(**labels, retrieval_strategy=retrieval_strategy(strategy)).set(score)
    if threshold is not None:
        RAG_EVALUATION_THRESHOLD.labels(**labels).set(threshold)
