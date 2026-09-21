"""Structured JSON logging with request/trace correlation and redaction."""
from __future__ import annotations

import json
import logging as std_logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from .redaction import digest_text, redact_event
from .request_id import get_request_id, get_run_id
from .tracing import current_trace_id

_FIELDS: ContextVar[dict[str, Any]] = ContextVar("telemetry_log_fields", default={})


def bind(**kwargs: Any) -> None:
    current = dict(_FIELDS.get())
    current.update(kwargs)
    _FIELDS.set(current)


def clear_context() -> None:
    _FIELDS.set({})


class JsonFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        message = record.getMessage()
        event: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event_name": message if message.isidentifier() and len(message) <= 64 else "log_message",
            "message_sha256": digest_text(message),
            "message_length": len(message),
            "request_id": get_request_id(),
            "run_id": get_run_id(),
            "trace_id": current_trace_id(),
            **_FIELDS.get(),
        }
        if record.exc_info:
            event["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "unknown"
        return json.dumps(redact_event(event), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def setup_logging(json: bool = True, *, level: int = std_logging.INFO) -> None:
    """Install one handler; JSON is the safe default for production logs."""
    root = std_logging.getLogger()
    root.setLevel(level)
    if not json:
        formatter: std_logging.Formatter = std_logging.Formatter("%(levelname)s %(name)s %(message)s")
    else:
        formatter = JsonFormatter()
    handler = std_logging.StreamHandler()
    handler.setFormatter(formatter)
    root.handlers[:] = [handler]
