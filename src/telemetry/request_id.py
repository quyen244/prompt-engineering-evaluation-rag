"""Request/run correlation without putting identifiers in metric labels."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import re
import uuid
from typing import Iterator

_REQUEST_ID: ContextVar[str | None] = ContextVar("telemetry_request_id", default=None)
_RUN_ID: ContextVar[str | None] = ContextVar("telemetry_run_id", default=None)
_TRACE_ID: ContextVar[str | None] = ContextVar("telemetry_trace_id", default=None)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def new_request_id() -> str:
    return str(uuid.uuid4())


def _checked(value: str | None, *, fallback: str) -> str:
    if value and _SAFE_ID.fullmatch(value):
        return value
    return fallback


def get_request_id() -> str | None:
    return _REQUEST_ID.get()


def get_run_id() -> str | None:
    return _RUN_ID.get()


def get_trace_id() -> str | None:
    return _TRACE_ID.get()


def set_request_id(value: str | None = None) -> str:
    value = _checked(value, fallback=new_request_id())
    _REQUEST_ID.set(value)
    return value


def set_run_id(value: str | None = None) -> str:
    value = _checked(value, fallback=new_request_id())
    _RUN_ID.set(value)
    return value


def set_trace_id(value: str | None = None) -> str:
    value = _checked(value, fallback=new_request_id())
    _TRACE_ID.set(value)
    return value


def clear_context() -> None:
    _REQUEST_ID.set(None)
    _RUN_ID.set(None)
    _TRACE_ID.set(None)


@contextmanager
def request_context(request_id: str | None = None, *, run_id: str | None = None) -> Iterator[str]:
    """Bind correlation IDs for one request and always restore prior context."""
    request_token = _REQUEST_ID.set(_checked(request_id, fallback=new_request_id()))
    run_token = _RUN_ID.set(_checked(run_id, fallback=new_request_id()))
    try:
        yield _REQUEST_ID.get() or ""
    finally:
        _REQUEST_ID.reset(request_token)
        _RUN_ID.reset(run_token)
