"""Versioned, privacy-safe telemetry event contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from .constants import (
    DATASET_SPLITS, ERROR_TYPES, EVENT_TYPES, GENERATION_FINISH_REASONS,
    MAX_EVENT_STRING_LENGTH, OUTCOMES, QUALITY_METRICS, RETRIEVAL_STRATEGIES,
)

_STRATEGY_ALIASES = {
    "bm25-only": "bm25_only", "bm25": "bm25_only", "bm25_only": "bm25_only",
    "bm25-plus-reranker": "bm25_reranker", "bm25_reranker": "bm25_reranker",
    "dense": "dense", "hybrid": "hybrid",
}
from .redaction import dumps_event, redact_event

SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _id() -> str:
    return str(uuid.uuid4())


def _validate_common(event_type: str, retrieval_strategy: str, dataset_split: str, outcome: str) -> None:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unsupported event_type: {event_type}")
    if retrieval_strategy not in RETRIEVAL_STRATEGIES:
        raise ValueError(f"unsupported retrieval_strategy: {retrieval_strategy}")
    if dataset_split not in DATASET_SPLITS:
        raise ValueError(f"unsupported dataset_split: {dataset_split}")
    if outcome not in OUTCOMES:
        raise ValueError(f"unsupported outcome: {outcome}")


def _measurement(name: str, value: Any, *, integer: bool = False) -> Any:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    if integer and int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value) if integer else value


def _score(name: str, value: float | None) -> None:
    if value is not None and not 0 <= _measurement(name, value) <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _bounded_identifier(name: str, value: str | None, *, required: bool = False) -> None:
    if value is None and not required:
        return
    if not isinstance(value, str) or not value or len(value) > MAX_EVENT_STRING_LENGTH:
        raise ValueError(f"{name} must be a non-empty string of at most {MAX_EVENT_STRING_LENGTH} characters")


def _validate_lineage(retrieval_strategy: str, prompt_version: str, model_version: str, dataset_split: str, outcome: str) -> str:
    try:
        canonical_strategy = _STRATEGY_ALIASES[retrieval_strategy.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unsupported retrieval_strategy: {retrieval_strategy}") from exc
    _validate_common("request", canonical_strategy, dataset_split, outcome)
    for name, value in (("prompt_version", prompt_version), ("model_version", model_version)):
        _bounded_identifier(name, value, required=True)
    return canonical_strategy


@dataclass(frozen=True)
class RequestEvent:
    """One request outcome; raw question/answer fields intentionally do not exist."""

    retrieval_strategy: str
    prompt_version: str
    model_version: str
    dataset_split: str = "production"
    run_id: str = field(default_factory=_id)
    request_id: str = field(default_factory=_id)
    trace_id: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    outcome: str = "success"
    event_type: str = field(default="request", init=False)
    event_version: str = field(default=SCHEMA_VERSION, init=False)
    occurred_at: str = field(default_factory=_now, init=False)

    def __post_init__(self) -> None:
        canonical_strategy = _validate_lineage(
            self.retrieval_strategy, self.prompt_version, self.model_version,
            self.dataset_split, self.outcome,
        )
        object.__setattr__(self, "retrieval_strategy", canonical_strategy)
        _measurement("latency_ms", self.latency_ms)
        _measurement("prompt_tokens", self.prompt_tokens, integer=True)
        _measurement("completion_tokens", self.completion_tokens, integer=True)
        _measurement("cost_usd", self.cost_usd)

    def to_dict(self) -> dict[str, Any]:
        return redact_event(asdict(self))

    def to_json(self) -> str:
        return dumps_event(asdict(self))


@dataclass(frozen=True)
class DocumentExperimentEvent:
    """Append-only document-level experiment record.

    ``document_id`` must be a stable non-content identifier (for example a corpus
    UUID), never a filename containing personal data or document text.
    """

    document_id: str
    operation: str
    retrieval_strategy: str
    prompt_version: str
    model_version: str
    dataset_split: str
    run_id: str = field(default_factory=_id)
    experiment_id: str = field(default_factory=_id)
    trace_id: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    outcome: str = "success"
    event_type: str = field(default="document_experiment", init=False)
    event_version: str = field(default=SCHEMA_VERSION, init=False)
    occurred_at: str = field(default_factory=_now, init=False)

    def __post_init__(self) -> None:
        canonical_strategy = _validate_lineage(
            self.retrieval_strategy, self.prompt_version, self.model_version,
            self.dataset_split, self.outcome,
        )
        object.__setattr__(self, "retrieval_strategy", canonical_strategy)
        for name, value, integer in (
            ("latency_ms", self.latency_ms, False),
            ("prompt_tokens", self.prompt_tokens, True),
            ("completion_tokens", self.completion_tokens, True),
            ("cost_usd", self.cost_usd, False),
        ):
            _measurement(name, value, integer=integer)
        if not self.document_id or self.operation not in {"ingest", "retrieve", "evaluate", "generate"}:
            raise ValueError("document_id and operation must use the document experiment vocabulary")

    def to_dict(self) -> dict[str, Any]:
        return redact_event(asdict(self))

    def to_json(self) -> str:
        return dumps_event(asdict(self))


@dataclass(frozen=True)
class RetrievalEvent:
    """A bounded retrieval-stage record; document contents and IDs are never included."""

    retrieval_strategy: str
    prompt_version: str
    model_version: str
    retrieved_document_count: int
    dataset_split: str = "production"
    request_id: str = field(default_factory=_id)
    run_id: str = field(default_factory=_id)
    trace_id: str | None = None
    top_score: float | None = None
    context_tokens: int | None = None
    latency_ms: float | None = None
    outcome: str = "success"
    event_type: str = field(default="retrieval", init=False)
    event_version: str = field(default=SCHEMA_VERSION, init=False)
    occurred_at: str = field(default_factory=_now, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieval_strategy", _validate_lineage(
            self.retrieval_strategy, self.prompt_version, self.model_version, self.dataset_split, self.outcome,
        ))
        _measurement("retrieved_document_count", self.retrieved_document_count, integer=True)
        _measurement("context_tokens", self.context_tokens, integer=True)
        _measurement("latency_ms", self.latency_ms)
        _score("top_score", self.top_score)

    def to_dict(self) -> dict[str, Any]:
        return redact_event(asdict(self))

    def to_json(self) -> str:
        return dumps_event(asdict(self))


@dataclass(frozen=True)
class GenerationEvent:
    """A generation-stage record with prompt/model lineage but no prompt or answer text."""

    retrieval_strategy: str
    prompt_version: str
    model_version: str
    dataset_split: str = "production"
    request_id: str = field(default_factory=_id)
    run_id: str = field(default_factory=_id)
    trace_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    finish_reason: str = "unknown"
    outcome: str = "success"
    event_type: str = field(default="generation", init=False)
    event_version: str = field(default=SCHEMA_VERSION, init=False)
    occurred_at: str = field(default_factory=_now, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieval_strategy", _validate_lineage(
            self.retrieval_strategy, self.prompt_version, self.model_version, self.dataset_split, self.outcome,
        ))
        for name, value, integer in (("prompt_tokens", self.prompt_tokens, True), ("completion_tokens", self.completion_tokens, True), ("cost_usd", self.cost_usd, False), ("latency_ms", self.latency_ms, False)):
            _measurement(name, value, integer=integer)
        if self.finish_reason not in GENERATION_FINISH_REASONS:
            raise ValueError(f"unsupported finish_reason: {self.finish_reason}")

    def to_dict(self) -> dict[str, Any]:
        return redact_event(asdict(self))

    def to_json(self) -> str:
        return dumps_event(asdict(self))


@dataclass(frozen=True)
class EvaluationEvent:
    """Offline evaluation result contract. Callers supply scores; this module never fabricates them."""

    retrieval_strategy: str
    prompt_version: str
    model_version: str
    quality_metric: str
    dataset_split: str
    quality_score: float | None = None
    quality_threshold: float | None = None
    request_id: str = field(default_factory=_id)
    run_id: str = field(default_factory=_id)
    trace_id: str | None = None
    latency_ms: float | None = None
    outcome: str = "success"
    event_type: str = field(default="evaluation", init=False)
    event_version: str = field(default=SCHEMA_VERSION, init=False)
    occurred_at: str = field(default_factory=_now, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieval_strategy", _validate_lineage(
            self.retrieval_strategy, self.prompt_version, self.model_version, self.dataset_split, self.outcome,
        ))
        if self.quality_metric not in QUALITY_METRICS:
            raise ValueError(f"unsupported quality_metric: {self.quality_metric}")
        _score("quality_score", self.quality_score)
        _score("quality_threshold", self.quality_threshold)
        _measurement("latency_ms", self.latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return redact_event(asdict(self))

    def to_json(self) -> str:
        return dumps_event(asdict(self))


@dataclass(frozen=True)
class ErrorEvent:
    """A safe error outcome: type/code only, never an exception message or stack trace."""

    retrieval_strategy: str
    prompt_version: str
    model_version: str
    stage: str
    error_type: str
    dataset_split: str = "production"
    request_id: str = field(default_factory=_id)
    run_id: str = field(default_factory=_id)
    trace_id: str | None = None
    error_code: str | None = None
    retryable: bool = False
    latency_ms: float | None = None
    outcome: str = "error"
    event_type: str = field(default="error", init=False)
    event_version: str = field(default=SCHEMA_VERSION, init=False)
    occurred_at: str = field(default_factory=_now, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieval_strategy", _validate_lineage(
            self.retrieval_strategy, self.prompt_version, self.model_version, self.dataset_split, self.outcome,
        ))
        if self.stage not in {"retrieve", "rerank", "generate", "query", "ingest"}:
            raise ValueError(f"unsupported error stage: {self.stage}")
        if self.error_type not in ERROR_TYPES:
            raise ValueError(f"unsupported error_type: {self.error_type}")
        _bounded_identifier("error_code", self.error_code)
        _measurement("latency_ms", self.latency_ms)

    def to_dict(self) -> dict[str, Any]:
        return redact_event(asdict(self))

    def to_json(self) -> str:
        return dumps_event(asdict(self))
