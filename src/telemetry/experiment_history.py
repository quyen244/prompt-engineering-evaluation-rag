"""Append-only JSONL history for document experiments."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from .events import DocumentExperimentEvent
from .metrics import mark_dropped
from .redaction import dumps_event


class ExperimentHistoryWriter:
    """Write immutable experiment records without ever logging document content.

    A failed history sink is a dropped telemetry signal, not a failed RAG request.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: DocumentExperimentEvent | dict[str, Any]) -> bool:
        try:
            if isinstance(event, DocumentExperimentEvent):
                payload = event.to_dict()
            else:
                payload = dict(event)
                if payload.get("event_type") != "document_experiment":
                    raise ValueError("history accepts document_experiment events only")
                # Apply redaction before writing externally supplied mappings.
                line = dumps_event(payload)
                payload = None
            if payload is not None:
                line = dumps_event(payload)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()
            return True
        except Exception:
            mark_dropped("event")
            return False
