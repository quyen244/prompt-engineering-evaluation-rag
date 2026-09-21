"""Compatibility entry point for the documented OpenTelemetry API."""
from .tracing import current_trace_id, setup_tracing, shutdown_tracing, span, tracer

__all__ = ["setup_tracing", "shutdown_tracing", "span", "tracer", "current_trace_id"]
