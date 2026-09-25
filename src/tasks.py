"""Background jobs for long-running RAG work."""
from __future__ import annotations

from pathlib import Path
import time

from src.celery_app import celery_app
from src.telemetry.metrics import record_error, record_ingestion


@celery_app.task(bind=True, name="src.tasks.ingest_pdf")
def ingest_pdf(self, staged_path: str, filename: str) -> dict:
    """Ingest a staged PDF using the same pipeline as the synchronous baseline."""
    source = Path(staged_path)
    try:
        from src.server import _ingest_pdf

        started = time.perf_counter()
        record = _ingest_pdf(filename, source.read_bytes())
        record_ingestion("success", chunk_count=record["chunkCount"])
        return {**record, "processingSeconds": round(time.perf_counter() - started, 6)}
    except (ValueError, RuntimeError):
        record_ingestion("error")
        record_error("ingest", "upstream")
        raise
    except Exception:
        record_ingestion("error")
        record_error("ingest", "unknown")
        raise
    finally:
        source.unlink(missing_ok=True)
