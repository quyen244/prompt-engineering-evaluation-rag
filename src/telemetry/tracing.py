"""Fail-open OpenTelemetry setup and stable RAG span contracts."""
from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any, Iterator

from .metrics import mark_dropped

_PROVIDER: Any = None
_TRACER: Any = None

try:  # Optional for CLI/document experiments that do not install OTel.
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    trace = None
    Status = StatusCode = None
    _OTEL_AVAILABLE = False


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan": return self
    def __exit__(self, *_: Any) -> None: return None
    def set_attribute(self, *_: Any, **__: Any) -> None: return None
    def record_exception(self, *_: Any, **__: Any) -> None: return None
    def set_status(self, *_: Any, **__: Any) -> None: return None
    def add_event(self, *_: Any, **__: Any) -> None: return None


class _NoopTracer:
    def start_as_current_span(self, *_: Any, **__: Any) -> _NoopSpan: return _NoopSpan()


def setup_tracing(service_name: str, endpoint: str | None = None) -> Any:
    """Configure a batch exporter. Any exporter failure leaves the app running."""
    global _PROVIDER, _TRACER
    if not _OTEL_AVAILABLE:
        _TRACER = _NoopTracer()
        return None
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        target = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if target:
            # grpc exporter accepts host:port; callers can configure TLS separately.
            exporter = OTLPSpanExporter(endpoint=target)
            provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=2048, max_export_batch_size=256))
        trace.set_tracer_provider(provider)
        _PROVIDER = provider
        _TRACER = provider.get_tracer("rag.telemetry", "1.0")
    except Exception:
        mark_dropped("span")
        _PROVIDER = None
        _TRACER = _NoopTracer()
    return _PROVIDER


def shutdown_tracing() -> None:
    global _PROVIDER, _TRACER
    provider, _PROVIDER = _PROVIDER, None
    _TRACER = _NoopTracer()
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            mark_dropped("span")


def tracer() -> Any:
    global _TRACER
    if _TRACER is None:
        _TRACER = trace.get_tracer("rag.telemetry", "1.0") if _OTEL_AVAILABLE else _NoopTracer()
    return _TRACER


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    # Span attributes have more room than metric labels, but still must not contain
    # user content or secrets. Only contract fields are copied.
    allowed = {
        "rag.retrieval.strategy", "rag.dataset.split", "rag.prompt.version",
        "rag.model.version", "rag.run.id", "rag.request.id", "rag.document.id",
        "rag.outcome", "rag.top_score", "rag.stage", "rag.error.type", "gen_ai.request.model",
        "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.response.finish_reasons",
    }
    result = {}
    for key, value in attributes.items():
        if key in allowed and isinstance(value, (str, int, float, bool, list, tuple)):
            result[key] = value
    return result


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Create a stable span while ensuring telemetry exceptions never escape."""
    try:
        context = tracer().start_as_current_span(name, attributes=_safe_attributes(attributes))
    except Exception:
        mark_dropped("span")
        yield _NoopSpan()
        return
    # Exceptions from the application body must remain application exceptions; only
    # failures while creating telemetry are swallowed.
    with context as current:
        yield current


def mark_span_error(current: Any, error: BaseException) -> None:
    try:
        # Do not record exception messages: they can contain user content or secrets.
        current.set_attribute("rag.error.type", type(error).__name__)
        if _OTEL_AVAILABLE and Status is not None and StatusCode is not None:
            current.set_status(Status(StatusCode.ERROR, type(error).__name__))
    except Exception:
        mark_dropped("span")


def current_trace_id() -> str | None:
    if not _OTEL_AVAILABLE:
        return None
    try:
        current = trace.get_current_span()
        context = current.get_span_context()
        if context and context.is_valid:
            return format(context.trace_id, "032x")
    except Exception:
        mark_dropped("span")
    return None
