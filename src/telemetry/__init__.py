"""Privacy-safe observability contracts for the RAG application."""

from .events import DocumentExperimentEvent, RequestEvent
from .experiment_history import ExperimentHistoryWriter
from .metrics import (
    CONTEXT_TOKENS,
    LLM_COST,
    LLM_TOKENS,
    RAG_DURATION,
    RAG_ERRORS,
    RAG_IN_FLIGHT,
    RAG_NO_ANSWER,
    RAG_TOP_SCORE,
    record_llm_call,
)
from .request_id import clear_context, get_request_id, new_request_id, request_context
from .tracing import setup_tracing, shutdown_tracing, span

__all__ = [
    "RequestEvent", "DocumentExperimentEvent", "ExperimentHistoryWriter",
    "setup_tracing", "shutdown_tracing", "span", "request_context",
    "new_request_id", "get_request_id", "clear_context", "RAG_DURATION",
    "RAG_ERRORS", "RAG_NO_ANSWER", "RAG_IN_FLIGHT", "RAG_TOP_SCORE",
    "LLM_TOKENS", "LLM_COST", "CONTEXT_TOKENS", "record_llm_call",
]
