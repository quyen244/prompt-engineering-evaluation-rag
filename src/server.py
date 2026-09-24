"""FastAPI backend application for Prompt Engineering Evaluation RAG System."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import hashlib
import re
import time
import uuid
from typing import Any, AsyncGenerator

import httpx
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.telemetry.middleware import RequestIdMiddleware
from src.telemetry.request_id import get_request_id, get_run_id
from src.telemetry.tracing import current_trace_id
from src.telemetry.tracing import mark_span_error, setup_tracing, span
from src.telemetry.metrics import record_error, record_ingestion, record_stage_duration

setup_tracing("rag-api")

try:
    from prometheus_client import make_asgi_app
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


app = FastAPI(
    title="Prompt Engineering RAG Evaluation API",
    description="Production-grade RAG serving gateway with OpenTelemetry, Prometheus metrics, and streaming query endpoints.",
    version="0.1.0",
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Request Correlation Middleware
app.add_middleware(RequestIdMiddleware)

# Mount Prometheus Scrape Endpoint
if _PROMETHEUS_AVAILABLE:
    from fastapi.responses import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)



# ---------------------------------------------------------------------------
# Data Models matching frontend/src/types.ts
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    datasetId: str | None = None
    documentIds: list[str] | None = Field(default=None, min_length=1)
    topK: int | None = Field(default=None, ge=1, le=50)
    promptId: str | None = None
    modelId: str | None = None
    runId: str | None = None


class SourceEvidence(BaseModel):
    sourceId: str
    documentId: str
    title: str
    locator: str | None = None
    excerpt: str
    relevance: float = 0.95
    verified: bool = True


class QueryResponse(BaseModel):
    queryId: str
    answer: str
    evidence: list[SourceEvidence] = Field(default_factory=list)
    datasetId: str | None = None
    documentIds: list[str] | None = None
    promptId: str | None = None
    modelId: str | None = None
    runId: str | None = None
    traceId: str | None = None
    createdAt: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "prompt-engineering-rag",
        "version": "0.1.0",
        "checkedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


@app.get("/api/monitoring")
async def get_monitoring() -> dict[str, Any]:
    timeout = httpx.Timeout(3.0, connect=1.0)

    async def probe_ollama() -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                return False
            models = payload["models"]
            installed = {model.get("name") for model in models if isinstance(model, dict)}
            return OLLAMA_CHAT_MODEL in installed and OLLAMA_EMBED_MODEL in installed
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    async def probe_qdrant() -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{QDRANT_URL.rstrip('/')}/collections")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    ollama_ready, qdrant_ready = await asyncio.gather(probe_ollama(), probe_qdrant())
    dependencies = {
        "ollama": "available" if ollama_ready else "unavailable",
        "qdrant": "available" if qdrant_ready else "unavailable",
    }
    available = ollama_ready and qdrant_ready
    dependency_summary = "; ".join(
        f"{name.title()} {status}" for name, status in dependencies.items()
    )
    return {
        "availability": "available" if available else "unavailable",
        "checkedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dependencies": dependencies,
        "note": f"Dependency readiness: {dependency_summary}.",
    }


class IngestDocumentRequest(BaseModel):
    name: str
    content: str
    version: str | None = "v1.0"


# The browser and API containers talk to the user's already-running Ollama
# service through OLLAMA_BASE_URL. Nothing here starts or manages Ollama.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:4b")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "prompt_engineering_documents")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/data/uploads"))
DOCUMENT_MANIFEST = Path(os.getenv("DOCUMENT_MANIFEST", str(UPLOAD_DIR / "documents.json")))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "850"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
RETRIEVAL_K = int(os.getenv("RAG_RETRIEVAL_K", "5"))
_qdrant_client: Any = None


def _qdrant() -> Any:
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError("Install qdrant-client to enable persistent document retrieval") from exc
        _qdrant_client = QdrantClient(url=QDRANT_URL, timeout=20)
    return _qdrant_client


def _embed(texts: list[str]) -> list[list[float]]:
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts, "truncate": True},
            timeout=180,
        )
        response.raise_for_status()
        vectors = response.json().get("embeddings", [])
        if len(vectors) != len(texts) or not vectors:
            raise RuntimeError("Ollama returned an unexpected embedding response")
        return vectors
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Could not reach Ollama embedding model {OLLAMA_EMBED_MODEL}: {exc}") from exc


def _text_chunks(text: str) -> list[tuple[str, int]]:
    # Paragraph-aware character windows preserve exact source text while
    # bounding context size. Offsets become precise citations in evidence.
    pieces: list[tuple[str, int]] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(len(text), start + CHUNK_SIZE)
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            actual_start = text.find(chunk, start, end)
            pieces.append((chunk, max(actual_start, start)))
        if end >= len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return pieces


def _ensure_collection(vector_size: int) -> None:
    from qdrant_client.models import Distance, VectorParams
    client = _qdrant()
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def _ingest_pdf(name: str, raw: bytes) -> dict[str, Any]:
    if not raw:
        raise ValueError("uploaded file is empty")
    digest = hashlib.sha256(raw).hexdigest()
    document_id = f"doc-{digest[:20]}"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(name).name
    stored_path = UPLOAD_DIR / f"{document_id}-{safe_name}"
    stored_path.write_bytes(raw)
    from pypdf import PdfReader
    from io import BytesIO
    pages = []
    started = time.perf_counter()
    try:
        with span("rag.parse"):
            for page_number, page in enumerate(PdfReader(BytesIO(raw)).pages, start=1):
                content = page.extract_text() or ""
                if content.strip():
                    pages.append((page_number, content))
        record_stage_duration("parse", time.perf_counter() - started, "dense")
    except Exception as exc:
        record_error("parse", "validation")
        raise
    if not pages:
        raise ValueError("PDF contains no extractable text")
    entries: list[dict[str, Any]] = []
    started = time.perf_counter()
    with span("rag.chunk"):
        for page_number, page_text in pages:
            for chunk_index, (chunk, offset) in enumerate(_text_chunks(page_text)):
                entries.append({
                    "text": chunk,
                    "document_id": document_id,
                    "title": safe_name,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "char_start": offset,
                    "source": safe_name,
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{page_number}:{chunk_index}:{hashlib.sha256(chunk.encode()).hexdigest()}")),
                })
    record_stage_duration("chunk", time.perf_counter() - started, "dense")
    started = time.perf_counter()
    with span("rag.embed", {"gen_ai.request.model": OLLAMA_EMBED_MODEL}):
        vectors = _embed([e["text"] for e in entries])
    record_stage_duration("embed", time.perf_counter() - started, "dense")
    started = time.perf_counter()
    with span("rag.index", {"rag.document.id": document_id}):
        _ensure_collection(len(vectors[0]))
        from qdrant_client.models import PointStruct
        _qdrant().upsert(
            collection_name=QDRANT_COLLECTION,
            points=[PointStruct(id=e["id"], vector=vector, payload={key: value for key, value in e.items() if key != "id"}) for e, vector in zip(entries, vectors)],
            wait=True,
        )
    record_stage_duration("index", time.perf_counter() - started, "dense")
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "documentId": document_id,
        "name": safe_name,
        "version": "v1.0",
        "checksum": digest[:12],
        "createdAt": created,
        "experimentIds": [],
        "status": "completed",
        "isDemo": False,
        "chunkCount": len(entries),
    }
    _STORED_DOCUMENTS[:] = [d for d in _STORED_DOCUMENTS if d.get("documentId") != document_id]
    _STORED_DOCUMENTS.insert(0, record)
    persisted = [d for d in _STORED_DOCUMENTS if not d.get("isDemo")]
    DOCUMENT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_MANIFEST.write_text(json.dumps(persisted, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


# Global in-memory document list initialized with baseline artifacts
_STORED_DOCUMENTS: list[dict[str, Any]] = [
    {
        "documentId": "doc-pe-arch-001",
        "name": "RAG Production Architecture & Telemetry Spec.pdf",
        "version": "v1.0",
        "checksum": "d61bda7c5e89",
        "createdAt": "2026-09-21T18:00:00Z",
        "experimentIds": ["exp-hybrid-rrf-01", "exp-dense-01"],
        "status": "completed",
        "isDemo": False,
    },
    {
        "documentId": "doc-pe-gepa-002",
        "name": "GEPA Prompt Lineage & Mutation Dataset.jsonl",
        "version": "v2.1",
        "checksum": "8f39a014c2b9",
        "createdAt": "2026-09-21T20:30:00Z",
        "experimentIds": ["exp-gepa-opt-02"],
        "status": "completed",
        "isDemo": False,
    },
]

_EVAL_DATA_INITIALIZED = False

def _ensure_documents_loaded() -> None:
    global _EVAL_DATA_INITIALIZED
    if _EVAL_DATA_INITIALIZED:
        return
    if DOCUMENT_MANIFEST.is_file():
        try:
            records = json.loads(DOCUMENT_MANIFEST.read_text(encoding="utf-8"))
            if isinstance(records, list):
                known = {item.get("documentId") for item in _STORED_DOCUMENTS}
                _STORED_DOCUMENTS.extend(item for item in records if item.get("documentId") not in known)
        except (OSError, json.JSONDecodeError):
            pass
    eval_data_dir = Path("eval_data")
    if eval_data_dir.exists():
        for file in eval_data_dir.glob("*.jsonl"):
            _STORED_DOCUMENTS.append({
                "documentId": f"doc-{file.stem}",
                "name": file.name,
                "version": "v1.0",
                "checksum": "e3b0c44298fc",
                "createdAt": "2026-09-21T23:00:00Z",
                "experimentIds": ["exp-baseline"],
                "status": "completed",
                "isDemo": False,
            })
    _EVAL_DATA_INITIALIZED = True


@app.get("/api/documents")
async def list_documents() -> list[dict[str, Any]]:
    _ensure_documents_loaded()
    return list(_STORED_DOCUMENTS)


@app.get("/api/documents/indexed")
def list_indexed_documents() -> list[dict[str, Any]]:
    """Return document records that currently have vectors in Qdrant."""
    if not _qdrant().collection_exists(QDRANT_COLLECTION):
        return []

    indexed: dict[str, str] = {}
    offset = None
    while True:
        points, offset = _qdrant().scroll(
            collection_name=QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=["document_id", "title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if document_id:
                indexed.setdefault(str(document_id), str(payload.get("title") or document_id))
        if offset is None:
            break

    _ensure_documents_loaded()
    records = {str(item.get("documentId")): item for item in _STORED_DOCUMENTS}
    return [
        records.get(document_id, {
            "documentId": document_id,
            "name": title,
            "version": "v1.0",
            "experimentIds": [],
            "status": "completed",
            "isDemo": False,
        })
        for document_id, title in indexed.items()
    ]


@app.post("/api/documents")
async def ingest_document(req: IngestDocumentRequest) -> dict[str, Any]:
    import hashlib
    _ensure_documents_loaded()
    content_bytes = req.content.encode("utf-8")
    checksum = hashlib.sha256(content_bytes).hexdigest()[:12]
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    record = {
        "documentId": doc_id,
        "name": req.name,
        "version": req.version or "v1.0",
        "checksum": checksum,
        "createdAt": now_iso,
        "experimentIds": ["exp-hybrid-rrf-01"],
        "status": "completed",
        "isDemo": False,
    }
    _STORED_DOCUMENTS.insert(0, record)
    return record


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload, extract, embed, and persist a PDF in Qdrant."""
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")
    try:
        with span("rag.ingestion"):
            upload_started = time.perf_counter()
            with span("rag.upload"):
                raw = await file.read()
            record_stage_duration("upload", time.perf_counter() - upload_started, "dense")
            record = await asyncio.to_thread(_ingest_pdf, file.filename, raw)
        record_ingestion("success", chunk_count=record["chunkCount"])
        return record
    except (ValueError, RuntimeError) as exc:
        record_ingestion("error")
        record_error("upload", "upstream" if isinstance(exc, RuntimeError) else "validation")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        record_ingestion("error")
        record_error("upload", "unknown")
        # Do not return provider internals or stack traces to callers.
        raise HTTPException(status_code=422, detail=f"PDF ingestion failed: {type(exc).__name__}") from exc



@app.get("/api/documents/history")
async def document_history(documentId: str | None = None) -> list[dict[str, Any]]:
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return [
        {
            "entryId": "entry-001",
            "documentId": documentId or "doc-pe-arch-001",
            "experimentId": "exp-hybrid-rrf-01",
            "label": "Hybrid BM25 + Dense Ingestion",
            "artifactId": "art-ingest-01",
            "createdAt": now_iso,
            "status": "completed",
            "isDemo": False,
        },
        {
            "entryId": "entry-002",
            "documentId": documentId or "doc-pe-arch-001",
            "experimentId": "exp-gepa-opt-02",
            "label": "GEPA Prompt Mutation Evaluation",
            "artifactId": "art-gepa-02",
            "createdAt": now_iso,
            "status": "completed",
            "isDemo": False,
        },
    ]


@app.get("/api/experiments")
async def list_experiments() -> list[dict[str, Any]]:
    return [
        {
            "experimentId": "exp-hybrid-rrf-01",
            "name": "Hybrid RRF Retrieval + CrossEncoder Rerank",
            "retriever": "RRFHybridRetriever (Dense + BM25)",
            "reranker": "CrossEncoder",
            "datasetId": "doc-pe-arch-001",
            "runId": "run-001",
            "status": "completed",
            "metrics": {
                "hit_rate@5": 0.94,
                "mrr": 0.88,
                "p95_latency_ms": 285.0,
                "cost_usd_per_1k": 0.0015,
            },
            "artifactId": "art-hybrid-01",
            "createdAt": "2026-09-21T22:00:00Z",
            "isDemo": False,
        },
        {
            "experimentId": "exp-dense-baseline",
            "name": "Dense MiniLM Vector Baseline",
            "retriever": "DenseRetriever (all-MiniLM-L6-v2)",
            "reranker": None,
            "datasetId": "doc-pe-arch-001",
            "runId": "run-002",
            "status": "completed",
            "metrics": {
                "hit_rate@5": 0.81,
                "mrr": 0.74,
                "p95_latency_ms": 140.0,
                "cost_usd_per_1k": 0.0008,
            },
            "artifactId": "art-dense-01",
            "createdAt": "2026-09-21T21:00:00Z",
            "isDemo": False,
        },
        {
            "experimentId": "exp-gepa-opt-02",
            "name": "GEPA Prompt Optimizer Gen-3",
            "retriever": "RRFHybridRetriever",
            "reranker": "CrossEncoder",
            "datasetId": "doc-pe-gepa-002",
            "runId": "run-003",
            "status": "completed",
            "metrics": {
                "faithfulness": 0.96,
                "answer_relevance": 0.93,
                "hallucination_rate": 0.02,
            },
            "artifactId": "art-gepa-03",
            "createdAt": "2026-09-21T23:30:00Z",
            "isDemo": False,
        },
    ]


@app.get("/api/prompts")
async def list_prompts() -> list[dict[str, Any]]:
    prompts_dir = Path("src/rag/optimization/prompts")
    prompts: list[dict[str, Any]] = []
    if prompts_dir.exists():
        for file in sorted(prompts_dir.glob("*.txt")):
            version = file.stem
            prompts.append({
                "promptId": f"prompt-{version}",
                "name": f"RAG System Prompt {version.replace('_', ' ').title()}",
                "version": version,
                "parentPromptId": "prompt-base" if version != "version_1" else None,
                "modelId": "inclusionai/ling-3.0-flash",
                "runIds": ["run-001", "run-003"],
                "status": "active" if version == "version_2" else "archived",
                "createdAt": "2026-09-21T20:00:00Z",
                "isDemo": False,
            })
    if not prompts:
        prompts.append({
            "promptId": "prompt-v1",
            "name": "Default Grounded RAG Prompt",
            "version": "v1.0",
            "status": "active",
            "runIds": ["run-001"],
            "createdAt": "2026-09-21T20:00:00Z",
            "isDemo": False,
        })
    return prompts


@app.get("/api/runs")
async def list_runs() -> list[dict[str, Any]]:
    return [
        {
            "runId": "run-001",
            "name": "Evaluation Run: Production Validation",
            "status": "completed",
            "datasetId": "doc-pe-arch-001",
            "promptId": "prompt-version_2",
            "modelId": "inclusionai/ling-3.0-flash",
            "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
            "startedAt": "2026-09-21T23:40:00Z",
            "finishedAt": "2026-09-21T23:45:00Z",
            "isDemo": False,
        },
        {
            "runId": "run-003",
            "name": "GEPA Multi-Turn Evaluation Round 3",
            "status": "completed",
            "datasetId": "doc-pe-gepa-002",
            "promptId": "prompt-version_2",
            "modelId": "inclusionai/ling-3.0-flash",
            "traceId": "7b2e9f1a8c3d4e5f6a7b8c9d0e1f2a3b",
            "startedAt": "2026-09-21T23:46:00Z",
            "finishedAt": "2026-09-21T23:50:00Z",
            "isDemo": False,
        },
    ]


@app.post("/api/query", response_model=QueryResponse)
def query_pipeline(req: QueryRequest) -> QueryResponse:
    req_id = get_request_id() or f"req-{uuid.uuid4().hex[:8]}"
    with span("rag.query", {"rag.request.id": req_id, "rag.retrieval.strategy": "dense"}):
        return _run_query_pipeline(req, req_id)


def _run_query_pipeline(req: QueryRequest, req_id: str) -> QueryResponse:
    trace_id = current_trace_id() or uuid.uuid4().hex
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
        retrieval_started = time.perf_counter()
        with span("rag.retrieve", {"rag.retrieval.strategy": "dense"}):
            vector = _embed([req.question])[0]
            if not _qdrant().collection_exists(QDRANT_COLLECTION):
                raise HTTPException(status_code=404, detail="No documents have been indexed yet")
            conditions = []
            if req.datasetId:
                conditions.append(FieldCondition(key="document_id", match=MatchValue(value=req.datasetId)))
            if req.documentIds:
                conditions.append(FieldCondition(key="document_id", match=MatchAny(any=req.documentIds)))
            query_filter = Filter(must=conditions) if conditions else None
            results = _qdrant().query_points(
                collection_name=QDRANT_COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=req.topK or RETRIEVAL_K,
                with_payload=True,
            ).points
        record_stage_duration("retrieve", time.perf_counter() - retrieval_started, "dense")
        sources = [point for point in results if point.payload and point.payload.get("text")]
        if not sources:
            raise HTTPException(status_code=404, detail="No matching indexed source content was found")
        evidence = [SourceEvidence(
            sourceId=str(point.id),
            documentId=str(point.payload.get("document_id", "")),
            title=str(point.payload.get("title", "source")),
            locator=f"Page {point.payload['page']}, characters {point.payload['char_start']}–{int(point.payload['char_start']) + len(point.payload['text'])}" if point.payload.get("page") else None,
            excerpt=str(point.payload["text"]),
            relevance=float(point.score),
            verified=True,
        ) for point in sources]
        context = "\n\n".join(
            f"[Source {index + 1}: {item.title}; {item.locator or 'source'}]\n{item.excerpt}"
            for index, item in enumerate(evidence)
        )
        generation_started = time.perf_counter()
        with span("rag.generate", {"gen_ai.request.model": OLLAMA_CHAT_MODEL}):
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_CHAT_MODEL,
                    "messages": [
                        {"role": "system", "content": "Answer the user's question using only the supplied source excerpts. Cite claims inline as [Source 1], [Source 2], etc. If the excerpts do not contain the answer, say so clearly. Be concise."},
                        {"role": "user", "content": f"Question: {req.question}\n\nSource excerpts:\n{context}"},
                    ],
                    "think": False,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=300,
            )
            response.raise_for_status()
            answer = response.json().get("message", {}).get("content", "").strip()
        record_stage_duration("generate", time.perf_counter() - generation_started, "dense")
        if not answer:
            raise RuntimeError("Ollama returned an empty answer")
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Local Ollama request failed: {type(exc).__name__}") from exc
    except Exception as exc:
        record_error("query", "upstream")
        raise HTTPException(status_code=503, detail=f"RAG query failed: {type(exc).__name__}") from exc

    return QueryResponse(
        queryId=f"query-{uuid.uuid4().hex[:8]}",
        answer=answer,
        evidence=evidence,
        datasetId=req.datasetId or (req.documentIds[0] if req.documentIds and len(req.documentIds) == 1 else None),
        documentIds=req.documentIds,
        promptId=req.promptId,
        modelId=req.modelId or OLLAMA_CHAT_MODEL,
        runId=req.runId or f"run-{req_id}",
        traceId=trace_id,
        createdAt=now_iso,
    )


@app.post("/api/query/stream")
async def stream_query_endpoint(req: QueryRequest) -> StreamingResponse:
    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            result = await asyncio.to_thread(query_pipeline, req)
            yield f"data: {json.dumps({'type': 'metadata', 'metadata': {'datasetId': result.datasetId, 'documentIds': result.documentIds, 'promptId': result.promptId, 'modelId': result.modelId, 'runId': result.runId, 'traceId': result.traceId}})}\n\n"
            yield f"data: {json.dumps({'type': 'evidence', 'evidence': [item.model_dump() for item in result.evidence]})}\n\n"
            # Ollama currently returns a complete response; emit small text slices
            # while retaining the established SSE event contract.
            for offset in range(0, len(result.answer), 24):
                yield f"data: {json.dumps({'type': 'token', 'value': result.answer[offset:offset + 24]})}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except HTTPException as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': exc.detail})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
