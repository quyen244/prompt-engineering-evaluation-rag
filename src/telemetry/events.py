"""Versioned, privacy-safe telemetry event contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from .constants import DATASET_SPLITS, EVENT_TYPES, OUTCOMES, RETRIEVAL_STRATEGIES

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
        try:
            canonical_strategy = _STRATEGY_ALIASES[self.retrieval_strategy.lower()]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"unsupported retrieval_strategy: {self.retrieval_strategy}") from exc
        object.__setattr__(self, "retrieval_strategy", canonical_strategy)
        _validate_common(self.event_type, self.retrieval_strategy, self.dataset_split, self.outcome)
        _measurement("latency_ms", self.latency_ms)
        _measurement("prompt_tokens", self.prompt_tokens, integer=True)
        _measurement("completion_tokens", self.completion_tokens, integer=True)
        _measurement("cost_usd", self.cost_usd)
        if not self.prompt_version or not self.model_version:
            raise ValueError("prompt_version and model_version are required")

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
        try:
            canonical_strategy = _STRATEGY_ALIASES[self.retrieval_strategy.lower()]
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"unsupported retrieval_strategy: {self.retrieval_strategy}") from exc
        object.__setattr__(self, "retrieval_strategy", canonical_strategy)
        _validate_common(self.event_type, self.retrieval_strategy, self.dataset_split, self.outcome)
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
