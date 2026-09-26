"""Prepare the isolated TechQA Qdrant collection used by the live API load run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "eval_data" / "corpus.jsonl"
QUESTIONS = ROOT / "eval_data" / "eval_questions.json"


def _request(base: str, path: str, *, method: str = "GET", body: object | None = None) -> object:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def _load_dataset() -> tuple[list[dict], list[dict], str, str]:
    documents = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    ids = [item["doc_id"] for item in documents]
    question_ids = [item["id"] for item in questions]
    if len(documents) != 100 or len(set(ids)) != 100:
        raise ValueError("TechQA corpus must contain exactly 100 unique document IDs")
    if len(questions) != 20 or len(set(question_ids)) != 20:
        raise ValueError("TechQA question file must contain exactly 20 unique questions")
    if not all(isinstance(item.get("text"), str) and item["text"].strip() and item.get("title") for item in documents):
        raise ValueError("Each TechQA document needs a nonempty title and text")
    if not all(item.get("gold_doc_id") in ids and item.get("question", "").strip() for item in questions):
        raise ValueError("Every TechQA question needs text and a gold document in the corpus")
    corpus_hash = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    questions_hash = hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()
    return documents, questions, corpus_hash, questions_hash


def _chunks(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    # Match the FastAPI paragraph-aware character windows.
    text = text.strip()
    pieces = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind("\n", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            pieces.append((chunk, max(text.find(chunk, start, end), start)))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return pieces


def _plan(documents: list[dict], corpus_hash: str, model: str, size: int, overlap: int) -> tuple[str, list[dict]]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap smaller than chunk size")
    key = f"{corpus_hash}:{model}:{size}:{overlap}"
    collection = "techqa_eval_" + hashlib.sha256(key.encode()).hexdigest()[:16]
    points = []
    for doc in documents:
        for index, (chunk, offset) in enumerate(_chunks(doc["text"], size, overlap)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{key}:{doc['doc_id']}:{index}:{hashlib.sha256(chunk.encode()).hexdigest()}"))
            points.append({
                "id": point_id,
                "payload": {
                    "document_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": chunk,
                    "char_start": offset,
                    "chunk_index": index,
                    "dataset_sha256": corpus_hash,
                    "embedding_model": model,
                },
            })
    return collection, points


def _existing_points(qdrant_url: str, collection: str) -> list[dict] | None:
    try:
        _request(qdrant_url, f"/collections/{collection}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    found = []
    offset = None
    while True:
        body: dict[str, object] = {"limit": 256, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        result = _request(qdrant_url, f"/collections/{collection}/points/scroll", method="POST", body=body)["result"]
        found.extend(result["points"])
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return found


def _verify(expected: list[dict], actual: list[dict]) -> None:
    expected_by_id = {item["id"]: item["payload"] for item in expected}
    actual_by_id = {str(item["id"]): item.get("payload") for item in actual}
    if expected_by_id != actual_by_id:
        missing = len(expected_by_id.keys() - actual_by_id.keys())
        extra = len(actual_by_id.keys() - expected_by_id.keys())
        raise ValueError(f"collection does not match the TechQA dataset (missing points={missing}, extra points={extra}); use a fresh collection configuration")


def _index(args: argparse.Namespace, collection: str, points: list[dict]) -> None:
    existing = _existing_points(args.qdrant_url, collection)
    expected_by_id = {item["id"]: item["payload"] for item in points}
    actual_by_id = {str(item["id"]): item.get("payload") for item in existing or []}
    if any(expected_by_id.get(point_id) != payload for point_id, payload in actual_by_id.items()):
        raise ValueError("existing collection contains points outside this TechQA indexing plan")
    remaining_points = [item for item in points if item["id"] not in actual_by_id]
    if not remaining_points:
        return
    tags = _request(args.ollama_url, "/api/tags")
    installed = {model.get("name") for model in tags.get("models", [])}
    if args.model not in installed:
        raise ValueError(f"Ollama model {args.model} is unavailable; pull it before indexing")
    first = _request(args.ollama_url, "/api/embed", method="POST", body={"model": args.model, "input": [remaining_points[0]["payload"]["text"]], "truncate": True})
    first_vector = first["embeddings"][0]
    if not first_vector:
        raise ValueError("Ollama returned an empty embedding")
    if existing is None:
        _request(args.qdrant_url, f"/collections/{collection}", method="PUT", body={"vectors": {"size": len(first_vector), "distance": "Cosine"}})
    for start in range(0, len(remaining_points), args.batch_size):
        batch = remaining_points[start:start + args.batch_size]
        if start == 0:
            vectors = [first_vector]
            remaining = batch[1:]
        else:
            vectors = []
            remaining = batch
        if remaining:
            result = _request(args.ollama_url, "/api/embed", method="POST", body={"model": args.model, "input": [item["payload"]["text"] for item in remaining], "truncate": True})
            vectors.extend(result["embeddings"])
        if len(vectors) != len(batch) or any(len(vector) != len(first_vector) for vector in vectors):
            raise ValueError("Ollama returned an unexpected embedding batch")
        with_vectors = [{**item, "vector": vector} for item, vector in zip(batch, vectors)]
        _request(args.qdrant_url, f"/collections/{collection}/points?wait=true", method="PUT", body={"points": with_vectors})
    _verify(points, _existing_points(args.qdrant_url, collection) or [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or verify an isolated TechQA collection for FastAPI load testing")
    parser.add_argument("mode", choices=("name", "check", "index"), help="name is offline; check reads Qdrant; index creates the collection if absent")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b"))
    parser.add_argument("--chunk-size", type=int, default=850)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        documents, questions, corpus_hash, questions_hash = _load_dataset()
        collection, points = _plan(documents, corpus_hash, args.model, args.chunk_size, args.chunk_overlap)
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        if args.mode == "index":
            _index(args, collection, points)
        elif args.mode == "check":
            _verify(points, _existing_points(args.qdrant_url, collection) or [])
        print(f"collection: {collection}")
        print(f"status: {args.mode} complete")
        print(f"documents: {len(documents)}")
        print(f"questions: {len(questions)}")
        print(f"points: {len(points)}")
        print(f"corpus_sha256: {corpus_hash}")
        print(f"questions_sha256: {questions_hash}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, urllib.error.URLError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        print("help: check eval_data files, Qdrant/Ollama URLs, and the configured embedding model")
        return 1


if __name__ == "__main__":
    sys.exit(main())
