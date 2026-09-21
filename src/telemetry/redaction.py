"""Privacy boundary for telemetry events.

Telemetry accepts metadata and measurements only. Raw user questions, generated answers,
credentials, and prompt bodies are never serialized. Questions may be represented by a
one-way digest plus length for duplicate detection.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .constants import SENSITIVE_KEY_PARTS

_SECRET_VALUE = re.compile(r"(?i)(bearer\s+|sk-[A-Za-z0-9_-]{12,}|api[_-]?key\s*[:=])")
_ALLOWED_RAW_TEXT_KEYS = {
    "event_type", "event_version", "outcome", "error_type", "stage",
    "retrieval_strategy", "prompt_version", "model_version", "dataset_split",
    "run_id", "request_id", "trace_id", "document_id", "experiment_id",
    "operation", "service_name", "event_name",
}


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key_name(key: object) -> str:
    return str(key).lower().replace("-", "_")


def _is_question(key: str) -> bool:
    return key in {"question", "query", "user_query", "raw_question", "question_text"}


def _is_answer(key: str) -> bool:
    return key in {"answer", "response", "raw_answer", "answer_text", "generated_text"}


def _is_secret(key: str) -> bool:
    return any(part in key for part in SENSITIVE_KEY_PARTS if part not in {"question", "answer", "prompt", "completion"})


def redact_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe, recursively redacted copy of an event."""
    output: dict[str, Any] = {}
    for original_key, value in event.items():
        key = str(original_key)
        normalized = _key_name(key)
        if _is_question(normalized):
            if isinstance(value, str):
                output["question_sha256"] = digest_text(value)
                output["question_length"] = len(value)
            else:
                output["question_redacted"] = True
            continue
        if _is_answer(normalized) or normalized in {"prompt", "completion", "prompt_text", "completion_text"}:
            output[f"{normalized}_length"] = len(value) if isinstance(value, str) else None
            continue
        if _is_secret(normalized):
            output[key] = "[REDACTED]"
            continue
        output[key] = _redact_value(normalized, value)
    return output


def _redact_value(parent_key: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_event(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(parent_key, item) for item in value]
    if isinstance(value, str):
        if parent_key not in _ALLOWED_RAW_TEXT_KEYS and _SECRET_VALUE.search(value):
            return "[REDACTED]"
        # Unknown free text is not useful in telemetry and can contain PII.
        if parent_key not in _ALLOWED_RAW_TEXT_KEYS and len(value) > 256:
            return {"sha256": digest_text(value), "length": len(value)}
    return value


def dumps_event(event: Mapping[str, Any]) -> str:
    """Canonical JSON serialization after applying the privacy boundary."""
    return json.dumps(redact_event(event), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
