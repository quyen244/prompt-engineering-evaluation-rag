"""Concurrent end-user query load test for pre-ingested RAG documents."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from gevent import spawn_later
from gevent.event import Event
from gevent.lock import Semaphore
import mlflow
from mlflow.tracking import MlflowClient
from locust import HttpUser, between, events, task
from locust.exception import StopUser

from scripts.load_test_support import (
    document_fingerprint,
    parse_server_timing,
    query_payload,
    summarize_stage_timings,
    summarize_stats,
)


BASE_URL = os.getenv("RAG_LOAD_BASE_URL", "http://localhost:8000").rstrip("/")
QUESTION = os.getenv(
    "RAG_LOAD_QUESTION",
    "What concepts and key points does this document explain?",
)
TOP_K = int(os.getenv("RAG_LOAD_TOP_K", "5"))
MAX_REQUESTS = int(os.getenv("RAG_LOAD_MAX_REQUESTS", "20"))
MAX_DURATION_SECONDS = int(os.getenv("RAG_LOAD_MAX_DURATION_SECONDS", "120"))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("RAG_LOAD_REQUEST_TIMEOUT_SECONDS", "60"))
if min(MAX_REQUESTS, MAX_DURATION_SECONDS, REQUEST_TIMEOUT_SECONDS) <= 0:
    raise ValueError("load-test request, duration, and timeout limits must be positive")
DOCUMENT_IDS: list[str] = []
TECHQA = os.getenv("RAG_LOAD_DATASET", "").casefold() == "techqa"
TECHQA_QUESTIONS: list[dict[str, str]] = []
TECHQA_CORPUS_SHA256 = ""
TECHQA_QUESTIONS_SHA256 = ""
TECHQA_COLLECTION = ""
STUDY_ID = uuid.uuid4().hex
RUN_STARTED_AT = 0.0
MLFLOW_RUN_STARTED = False
# Locust monkey-patches with gevent, so MLflow's thread-local active-run stack is
# not shared between the test_start and test_stop greenlets. Every write goes
# through an explicit client and run id or it lands in a stray auto-created run.
MLFLOW_RUN_ID = ""
MLFLOW_CLIENT: "MlflowClient | None" = None
TARGET_USERS = 0
REQUESTS_STARTED = 0
REQUESTS_COMPLETED = 0
STOP_REASON: str | None = None
RUN_TIMER = None
REQUEST_LOCK = Semaphore()
READY_USERS = 0
READY_EVENT = Event()
STAGE_SAMPLES: dict[str, list[float]] = {
    "embedding": [],
    "retrieval": [],
    "generation": [],
}


def _get_json(path: str) -> object:
    request = urllib.request.Request(f"{BASE_URL}{path}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"Load-test preflight failed for {path}: {type(exc).__name__}") from exc


def _log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        MLFLOW_CLIENT.log_param(MLFLOW_RUN_ID, key, value)


def _stop_at_duration(runner) -> None:
    global STOP_REASON
    with REQUEST_LOCK:
        if STOP_REASON is None:
            STOP_REASON = "duration_cap"
    if getattr(runner, "state", None) not in {"stopped", "stopping"}:
        runner.quit()


@events.test_start.add_listener
def prepare_study(environment, **_kwargs) -> None:
    global MLFLOW_RUN_STARTED
    try:
        _prepare_study(environment)
    except Exception as exc:
        environment.process_exit_code = 2
        print(f"error: load-test preflight failed: {exc}")
        if MLFLOW_CLIENT is not None and MLFLOW_RUN_ID:
            MLFLOW_CLIENT.set_tag(MLFLOW_RUN_ID, "preflight_error", type(exc).__name__)
            MLFLOW_CLIENT.set_terminated(MLFLOW_RUN_ID, status="FAILED")
            MLFLOW_RUN_STARTED = False
        spawn_later(0, environment.runner.quit)


def _prepare_study(environment) -> None:
    global BASE_URL, DOCUMENT_IDS, RUN_STARTED_AT, MLFLOW_RUN_STARTED, TARGET_USERS, STUDY_ID
    global REQUESTS_STARTED, REQUESTS_COMPLETED, STOP_REASON, RUN_TIMER, READY_USERS
    global TECHQA_QUESTIONS, TECHQA_CORPUS_SHA256, TECHQA_QUESTIONS_SHA256, TECHQA_COLLECTION
    global MLFLOW_RUN_ID, MLFLOW_CLIENT
    if getattr(environment.runner, "worker", False):
        return
    if MLFLOW_RUN_STARTED:
        return
    STUDY_ID = uuid.uuid4().hex
    with REQUEST_LOCK:
        REQUESTS_STARTED = 0
        REQUESTS_COMPLETED = 0
        STOP_REASON = None
        TARGET_USERS = 0
        READY_USERS = 0
        READY_EVENT.clear()
        for values in STAGE_SAMPLES.values():
            values.clear()
    BASE_URL = (getattr(environment, "host", None) or BASE_URL).rstrip("/")

    health = _get_json("/api/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError("RAG API health check did not return status=ok")

    indexed = _get_json("/api/documents/indexed")
    if not isinstance(indexed, list):
        raise RuntimeError("Indexed-document endpoint returned an unexpected response")
    available = [
        str(item["documentId"])
        for item in indexed
        if isinstance(item, dict) and item.get("documentId")
    ]
    requested = [value.strip() for value in os.getenv("RAG_LOAD_DOCUMENT_IDS", "").split(",") if value.strip()]
    if TECHQA:
        from scripts.techqa_index import _load_dataset, _plan

        documents, questions, corpus_hash, questions_hash = _load_dataset()
        collection, _ = _plan(
            documents, corpus_hash,
            os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
            int(os.getenv("RAG_CHUNK_SIZE", "850")),
            int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        )
        expected = {item["doc_id"] for item in documents}
        if set(available) != expected or len(available) != len(expected):
            raise RuntimeError("API indexed documents differ from the exact 100-document TechQA corpus; check QDRANT_COLLECTION")
        if requested and set(requested) != expected:
            raise RuntimeError("RAG_LOAD_DOCUMENT_IDS must include all 100 TechQA documents")
        requested = sorted(expected)
        TECHQA_QUESTIONS = questions
        TECHQA_CORPUS_SHA256 = corpus_hash
        TECHQA_QUESTIONS_SHA256 = questions_hash
        TECHQA_COLLECTION = collection
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise RuntimeError(f"Requested document IDs are not currently indexed ({len(missing)} missing)")
        DOCUMENT_IDS = list(dict.fromkeys(requested))
    else:
        DOCUMENT_IDS = available
    if not DOCUMENT_IDS:
        raise RuntimeError("No indexed documents found; ingest a document before starting Locust")
    query_payload(QUESTION, DOCUMENT_IDS, TOP_K)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("RAG_LOAD_EXPERIMENT", "rag-query-concurrency")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    MLFLOW_CLIENT = MlflowClient(tracking_uri=tracking_uri)
    run = MLFLOW_CLIENT.create_run(
        experiment.experiment_id,
        run_name=os.getenv("RAG_LOAD_RUN_NAME") or f"query-load-{STUDY_ID[:8]}",
        tags={
            "run_kind": "rag_load_test",
            "study_id": STUDY_ID,
            "request_scope": "full_rag_query_without_ingestion",
            "retrieval_strategy": "dense",
            "serving_configuration": "fastapi_qdrant_dense_ollama",
        },
    )
    MLFLOW_RUN_ID = run.info.run_id
    MLFLOW_RUN_STARTED = True
    options = getattr(environment, "parsed_options", None)
    TARGET_USERS = int(getattr(options, "num_users", 0) or 0)
    if TECHQA and TARGET_USERS not in (1, 20, 100):
        raise RuntimeError("TechQA load profile requires exactly 1, 20, or 100 users")
    if TECHQA and MAX_REQUESTS != (100 if TARGET_USERS == 100 else 20):
        raise RuntimeError("TechQA load profile requires 20 requests for 1 or 20 users, or 100 requests for 100 users")
    if TECHQA and os.getenv("QDRANT_COLLECTION") != TECHQA_COLLECTION:
        raise RuntimeError(f"Set QDRANT_COLLECTION={TECHQA_COLLECTION} for the TechQA load profile")
    _log_params({
        "base_url": BASE_URL,
        "document_count": len(DOCUMENT_IDS),
        "document_ids_sha256": document_fingerprint(DOCUMENT_IDS),
        "question_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
        "top_k": TOP_K,
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b"),
        "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"),
        "spawn_rate_per_second": float(getattr(options, "spawn_rate", 0) or 0),
        "max_requests": MAX_REQUESTS,
        "max_duration_seconds": MAX_DURATION_SECONDS,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "dataset": "techqa" if TECHQA else "indexed_documents",
        "retrieval_strategy": "dense",
        **({
            "corpus_sha256": TECHQA_CORPUS_SHA256,
            "questions_sha256": TECHQA_QUESTIONS_SHA256,
            "qdrant_collection": TECHQA_COLLECTION,
            "question_count": len(TECHQA_QUESTIONS),
        } if TECHQA else {}),
    })
    RUN_STARTED_AT = time.monotonic()
    RUN_TIMER = spawn_later(MAX_DURATION_SECONDS, _stop_at_duration, environment.runner)


@events.test_stop.add_listener
def record_study(environment, **_kwargs) -> None:
    global MLFLOW_RUN_STARTED, RUN_TIMER
    if not MLFLOW_RUN_STARTED or getattr(environment.runner, "worker", False):
        return
    elapsed = time.monotonic() - RUN_STARTED_AT if RUN_STARTED_AT else 0.0
    stats = environment.runner.stats.total
    results = summarize_stats(
        stats,
        duration_seconds=elapsed,
        user_count=TARGET_USERS or environment.runner.user_count,
    )
    _log_params({"target_users": int(results["user_count"])})
    results.update(summarize_stage_timings(STAGE_SAMPLES))
    results["requests_started"] = float(REQUESTS_STARTED)
    stop_reason = STOP_REASON or "manual"
    complete = REQUESTS_COMPLETED == MAX_REQUESTS and int(results["request_count"]) == MAX_REQUESTS
    successful = complete and int(results["failure_count"]) == 0
    if not successful:
        environment.process_exit_code = 1
    MLFLOW_CLIENT.set_tag(MLFLOW_RUN_ID, "stop_reason", stop_reason)
    MLFLOW_CLIENT.set_tag(MLFLOW_RUN_ID, "complete", str(complete).lower())
    for key, value in results.items():
        MLFLOW_CLIENT.log_metric(MLFLOW_RUN_ID, key, float(value))
    MLFLOW_CLIENT.log_dict(MLFLOW_RUN_ID, {
        "study_id": STUDY_ID,
        "request_scope": "POST /api/query: embedding + retrieval + Ollama generation; no uploads",
        "document_count": len(DOCUMENT_IDS),
        "document_ids_sha256": document_fingerprint(DOCUMENT_IDS),
        "stop_reason": stop_reason,
        "dataset": "techqa" if TECHQA else "indexed_documents",
        "corpus_sha256": TECHQA_CORPUS_SHA256 if TECHQA else None,
        "questions_sha256": TECHQA_QUESTIONS_SHA256 if TECHQA else None,
        "qdrant_collection": TECHQA_COLLECTION if TECHQA else None,
        **results,
    }, "load-test/summary.json")
    # MLflow prints a run link containing an emoji on end_run. On Windows,
    # PowerShell can expose a cp1252 stdout that raises after metrics upload.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    MLFLOW_CLIENT.set_terminated(MLFLOW_RUN_ID, status="FINISHED" if successful else "FAILED")
    print(f"mlflow_run_id: {MLFLOW_RUN_ID} status: {'FINISHED' if successful else 'FAILED'}")
    MLFLOW_RUN_STARTED = False
    if RUN_TIMER is not None and not RUN_TIMER.dead:
        RUN_TIMER.kill(block=False)
    RUN_TIMER = None


class RagQueryUser(HttpUser):
    host = BASE_URL
    wait_time = between(1, 3)

    def on_start(self) -> None:
        global TARGET_USERS, READY_USERS
        if not TECHQA:
            TARGET_USERS = max(TARGET_USERS, int(self.environment.runner.user_count))
            return
        with REQUEST_LOCK:
            READY_USERS += 1
            if READY_USERS >= TARGET_USERS:
                READY_EVENT.set()
        if not READY_EVENT.wait(timeout=MAX_DURATION_SECONDS):
            raise StopUser()

    @task
    def query_indexed_documents(self) -> None:
        global REQUESTS_STARTED, REQUESTS_COMPLETED, STOP_REASON
        with REQUEST_LOCK:
            if REQUESTS_STARTED >= MAX_REQUESTS:
                if TECHQA:
                    raise StopUser()
                return
            request_index = REQUESTS_STARTED
            REQUESTS_STARTED += 1
        question = TECHQA_QUESTIONS[request_index % len(TECHQA_QUESTIONS)]["question"] if TECHQA else QUESTION
        body = query_payload(question, DOCUMENT_IDS, TOP_K)
        try:
            with self.client.post(
                "/api/query",
                json=body,
                name="POST /api/query",
                timeout=REQUEST_TIMEOUT_SECONDS,
                catch_response=True,
            ) as response:
                timings = parse_server_timing(response.headers.get("Server-Timing", ""))
                with REQUEST_LOCK:
                    for stage, duration_ms in timings.items():
                        if stage in STAGE_SAMPLES:
                            STAGE_SAMPLES[stage].append(duration_ms)
                if response.status_code != 200:
                    response.failure(f"API returned HTTP {response.status_code}")
                    return
                try:
                    payload = response.json()
                except ValueError:
                    response.failure("API returned invalid JSON")
                    return
                if not isinstance(payload, dict) or not payload.get("answer"):
                    response.failure("API response did not contain an answer")
        finally:
            with REQUEST_LOCK:
                REQUESTS_COMPLETED += 1
                should_stop = (
                    REQUESTS_STARTED >= MAX_REQUESTS
                    and REQUESTS_COMPLETED >= MAX_REQUESTS
                )
                if should_stop and STOP_REASON is None:
                    STOP_REASON = "request_cap"
            if should_stop:
                # quit() joins the greenlet pool, so calling it from this task's
                # own greenlet deadlocks and test_stop never records the run.
                spawn_later(0, self.environment.runner.quit)
        # One request per user only when the user count already covers the cap
        # (20 users x 1, 100 users x 1). A single user must keep looping to
        # reach a 20-request cap on its own.
        if TECHQA and TARGET_USERS >= MAX_REQUESTS:
            raise StopUser()
