"""Append-only experiment and document provenance history."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def append_event(path: str | Path, event_type: str, payload: Mapping[str, Any]) -> None:
    """Append one immutable JSONL event; never rewrite or truncate history."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"event_type": event_type, "recorded_at": _now(), **dict(payload)}
    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    with _LOCK:
        with destination.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def record_document(path: str | Path, *, run_id: str, case_id: str, document_id: str, rank: int, score: float | None) -> None:
    append_event(path, "document_retrieved", {
        "run_id": run_id, "case_id": case_id, "document_id": document_id,
        "rank": rank, "score": score,
    })


def record_run(path: str | Path, *, run_id: str, mode: str, config: Mapping[str, Any], metrics: Mapping[str, float], case_count: int) -> None:
    append_event(path, "run_completed", {
        "run_id": run_id, "mode": mode, "config": dict(config),
        "metrics": dict(metrics), "case_count": case_count,
    })
