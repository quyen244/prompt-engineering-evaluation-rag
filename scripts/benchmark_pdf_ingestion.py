"""Measure the same PDF through synchronous and Celery ingestion, recording all trials in MLflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable


SyncIngest = Callable[[str, bytes], dict[str, Any]]
AsyncIngest = Callable[[str, bytes, float, float], dict[str, Any]]


def _sync_ingest(filename: str, content: bytes) -> dict[str, Any]:
    from src.server import _ingest_pdf

    started = time.perf_counter()
    result = _ingest_pdf(filename, content)
    return {"end_to_end_seconds": time.perf_counter() - started, "result": result}


def _celery_ingest(filename: str, content: bytes, poll_seconds: float, timeout_seconds: float) -> dict[str, Any]:
    from src.celery_app import celery_app
    from celery.result import AsyncResult
    from src.server import UPLOAD_DIR

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"benchmark-{os.getpid()}-{time.time_ns()}"
    staged_path = UPLOAD_DIR / f"{job_id}.pdf"
    staged_path.write_bytes(content)
    started = time.perf_counter()
    try:
        celery_app.send_task("src.tasks.ingest_pdf", args=[str(staged_path), filename], task_id=job_id)
        enqueued = time.perf_counter()
        task = AsyncResult(job_id, app=celery_app)
        while not task.ready():
            if time.perf_counter() - started > timeout_seconds:
                raise TimeoutError(f"Celery ingestion exceeded {timeout_seconds} seconds")
            time.sleep(poll_seconds)
        if task.failed():
            # Avoid serializing provider errors or stack traces into MLflow artifacts.
            raise RuntimeError(f"Celery ingestion failed ({type(task.result).__name__})")
        result = task.result
        return {
            "enqueue_seconds": enqueued - started,
            "end_to_end_seconds": time.perf_counter() - started,
            "worker_processing_seconds": float(result["processingSeconds"]),
            "result": result,
        }
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise


def run_benchmark(
    *, pdf_path: str | Path, trials: int, experiment_name: str, tracking_uri: str,
    sync_ingest: SyncIngest, async_ingest: AsyncIngest, mlflow_module: Any,
    poll_seconds: float = 0.5, timeout_seconds: float = 1800,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    path = Path(pdf_path)
    content = path.read_bytes()
    if not content.startswith(b"%PDF"):
        raise ValueError("input file does not have a PDF header")
    digest = hashlib.sha256(content).hexdigest()
    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment(experiment_name)

    rows: list[dict[str, Any]] = []
    with mlflow_module.start_run(run_name=f"pdf-ingestion-{path.stem}") as parent:
        parent_id = parent.info.run_id
        mlflow_module.set_tags({"run_kind": "ingestion_benchmark", "measurement": "real_pdf_sync_vs_celery"})
        mlflow_module.log_params({
            "pdf_name": path.name,
            "pdf_sha256": digest,
            "pdf_bytes": len(content),
            "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
            "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b"),
            "qdrant_collection": os.getenv("QDRANT_COLLECTION", "prompt_engineering_documents"),
            "trial_count": trials,
            "poll_interval_seconds": poll_seconds,
            "worker_concurrency": os.getenv("CELERY_WORKER_CONCURRENCY", "1"),
        })

        for index in range(trials):
            order = ("sync", "celery") if index % 2 == 0 else ("celery", "sync")
            pair: dict[str, dict[str, Any]] = {}
            for mode in order:
                measured = sync_ingest(path.name, content) if mode == "sync" else async_ingest(
                    path.name, content, poll_seconds, timeout_seconds,
                )
                record = measured.pop("result", {})
                safe_result = {
                    "document_id": record.get("documentId"),
                    "chunk_count": record.get("chunkCount"),
                }
                pair[mode] = {**measured, **safe_result}
                with mlflow_module.start_run(run_name=f"{mode}-trial-{index + 1}", nested=True):
                    mlflow_module.set_tags({"mode": mode, "pair_index": str(index + 1), "pdf_sha256": digest})
                    mlflow_module.log_params({"trial": index + 1, "mode": mode})
                    mlflow_module.log_metrics({key: value for key, value in measured.items() if isinstance(value, (int, float))})
                    mlflow_module.log_dict(pair[mode], f"trial-{index + 1}/{mode}.json")
            rows.append({"trial": index + 1, "order": list(order), "sync": pair["sync"], "celery": pair["celery"]})

        sync_times = [row["sync"]["end_to_end_seconds"] for row in rows]
        celery_times = [row["celery"]["end_to_end_seconds"] for row in rows]
        summary = {
            "parent_run_id": parent_id,
            "pdf_sha256": digest,
            "trial_count": trials,
            "sync_mean_seconds": statistics.mean(sync_times),
            "sync_median_seconds": statistics.median(sync_times),
            "celery_mean_end_to_end_seconds": statistics.mean(celery_times),
            "celery_median_end_to_end_seconds": statistics.median(celery_times),
            "mean_end_to_end_delta_seconds": statistics.mean(celery_times) - statistics.mean(sync_times),
            "mean_end_to_end_delta_percent": ((statistics.mean(celery_times) / statistics.mean(sync_times)) - 1) * 100 if statistics.mean(sync_times) else None,
            "mean_enqueue_seconds": statistics.mean(row["celery"]["enqueue_seconds"] for row in rows),
            "mean_worker_processing_seconds": statistics.mean(row["celery"]["worker_processing_seconds"] for row in rows),
            "trials": rows,
        }
        mlflow_module.log_metrics({key: value for key, value in summary.items() if key.endswith(("seconds", "percent")) and isinstance(value, (int, float))})
        mlflow_module.log_dict(summary, "comparison/summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired sync-vs-Celery PDF ingestion benchmark tracked in MLflow.")
    parser.add_argument("--pdf", required=True, help="Path to the real PDF, visible in this runtime")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--experiment", default="rag-pdf-ingestion-ablation")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=1800)
    args = parser.parse_args()
    try:
        import mlflow
        summary = run_benchmark(
            pdf_path=args.pdf, trials=args.trials, experiment_name=args.experiment,
            tracking_uri=args.tracking_uri, sync_ingest=_sync_ingest,
            async_ingest=_celery_ingest, mlflow_module=mlflow,
            poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds,
        )
    except Exception as error:
        raise SystemExit(f"Benchmark failed; no comparison was reported: {type(error).__name__}: {error}") from error
    print(json.dumps({key: value for key, value in summary.items() if key != "trials"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
