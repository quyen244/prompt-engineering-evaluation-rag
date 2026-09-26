"""Pure helpers shared by the Locust scenario and its offline tests."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def document_fingerprint(document_ids: list[str]) -> str:
    payload = json.dumps(sorted(document_ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def query_payload(question: str, document_ids: list[str], top_k: int) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("question must not be empty")
    if not document_ids:
        raise ValueError("at least one indexed document is required")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    return {"question": question, "documentIds": document_ids, "topK": top_k}


def summarize_stats(stats: Any, *, duration_seconds: float, user_count: int) -> dict[str, Any]:
    requests = int(stats.num_requests)
    failures = int(stats.num_failures)
    percentile = stats.get_response_time_percentile
    return {
        "request_count": requests,
        "failure_count": failures,
        "failure_rate": failures / requests if requests else 0.0,
        "requests_per_second": requests / duration_seconds if duration_seconds > 0 else 0.0,
        "user_count": int(user_count),
        "duration_seconds": float(duration_seconds),
        "response_time_mean_ms": float(stats.avg_response_time or 0.0),
        "response_time_min_ms": float(stats.min_response_time or 0.0),
        "response_time_max_ms": float(stats.max_response_time or 0.0),
        "response_time_p50_ms": float(percentile(0.50) or 0.0),
        "response_time_p95_ms": float(percentile(0.95) or 0.0),
        "response_time_p99_ms": float(percentile(0.99) or 0.0),
    }


def parse_server_timing(header: str) -> dict[str, float]:
    """Parse numeric Server-Timing durations in milliseconds."""
    parsed: dict[str, float] = {}
    for entry in header.split(","):
        name, *parameters = entry.strip().split(";")
        for parameter in parameters:
            key, separator, value = parameter.strip().partition("=")
            if key == "dur" and separator:
                try:
                    duration = float(value)
                except ValueError:
                    continue
                if math.isfinite(duration) and duration >= 0:
                    parsed[name.strip()] = duration
    return parsed


def summarize_stage_timings(samples: dict[str, list[float]]) -> dict[str, float]:
    """Return bounded stage latency summaries for successful API responses."""
    summary: dict[str, float] = {}
    for stage in ("embedding", "retrieval", "generation"):
        values = sorted(float(value) for value in samples.get(stage, []) if math.isfinite(float(value)) and value >= 0)
        if not values:
            continue
        count = len(values)
        summary[f"stage_{stage}_sample_count"] = float(count)
        summary[f"stage_{stage}_mean_ms"] = sum(values) / count
        summary[f"stage_{stage}_p50_ms"] = values[max(0, math.ceil(0.50 * count) - 1)]
        summary[f"stage_{stage}_p95_ms"] = values[max(0, math.ceil(0.95 * count) - 1)]
    return summary
