"""Framework-neutral stage instrumentation."""
from __future__ import annotations

from functools import wraps
import time
from typing import Any, Callable, TypeVar

from .constants import RETRIEVAL_STAGES
from .metrics import record_stage_duration
from .tracing import mark_span_error, span

F = TypeVar("F", bound=Callable[..., Any])


def traced(stage: str, *, strategy: str = "hybrid") -> Callable[[F], F]:
    if stage not in RETRIEVAL_STAGES:
        raise ValueError(f"unsupported stage: {stage}")

    def decorate(function: F) -> F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            with span(f"rag.{stage}", {"rag.retrieval.strategy": strategy, "rag.stage": stage}) as current:
                try:
                    return function(*args, **kwargs)
                except Exception as exc:
                    mark_span_error(current, exc)
                    raise
                finally:
                    # Measuring in finally includes time spent on failed requests.
                    elapsed = time.perf_counter() - started
                    try:
                        record_stage_duration(stage, elapsed, strategy)
                    except Exception:
                        # Metrics are never allowed to break the product operation.
                        pass
        return wrapped  # type: ignore[return-value]
    return decorate
