"""In-memory dense retrieval factory for local Ollama and Hugging Face embeddings."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
from typing import Any

from src.eval.runner import EvalConfig

_INDEX_CACHE: dict[tuple[str, str], tuple[list[str], list[str], Any, Any]] = {}
_CORPUS_CACHE: dict[str, tuple[list[str], list[str], str]] = {}
_QUERY_CACHE: dict[tuple[str, str], Any] = {}


def _corpus(root: Path) -> tuple[list[str], list[str], str]:
    cache_key = str(root)
    if cache_key in _CORPUS_CACHE:
        return _CORPUS_CACHE[cache_key]
    excluded = {
        ".git", ".venv", "venv", "__pycache__", "node_modules", "mlruns", "eval_data",
        ".cache", ".hf_cache", ".pytest_cache", "uploads", "storage", "data", "build", "dist",
    }
    suffixes = {".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".sh", ".bat"}
    ids: list[str] = []
    texts: list[str] = []
    for directory, child_dirs, files in os.walk(root):
        child_dirs[:] = sorted(name for name in child_dirs if name not in excluded and not name.startswith("."))
        relative_dir = Path(directory).relative_to(root)
        for name in sorted(files):
            path = Path(directory) / name
            if (path.suffix.lower() not in suffixes or name.startswith(".env") or
                    name.endswith("lock.json") or path.stat().st_size > 1_000_000):
                continue
            source = (relative_dir / name).as_posix()
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                continue
            if text:
                ids.append(f"project:{source}")
                texts.append(text)
    if not texts:
        raise ValueError(f"no project source files found under {root}")
    fingerprint = hashlib.sha256(json.dumps(
        [{"document_id": document_id, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
         for document_id, text in zip(ids, texts)],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    result = (ids, texts, fingerprint)
    _CORPUS_CACHE[cache_key] = result
    return result


def _chunks(ids: list[str], texts: list[str], size: int = 1400, overlap: int = 200) -> tuple[list[str], list[str]]:
    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    for document_id, text in zip(ids, texts):
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            chunk = text[start:end].strip()
            if chunk:
                chunk_ids.append(document_id)
                chunk_texts.append(chunk)
            if end == len(text):
                break
            start = max(start + 1, end - overlap)
    return chunk_ids, chunk_texts


def _embedding_client(embedding: str, options: dict[str, Any]) -> Any:
    if embedding.startswith("ollama:"):
        import httpx

        model = embedding.removeprefix("ollama:")
        base_url = str(options.get("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))).rstrip("/")

        class OllamaEmbedder:
            def encode(self, texts: list[str]) -> list[list[float]]:
                response = httpx.post(f"{base_url}/api/embed", json={"model": model, "input": texts, "truncate": True}, timeout=300)
                response.raise_for_status()
                vectors = response.json().get("embeddings", [])
                if len(vectors) != len(texts):
                    raise RuntimeError("Ollama embedding response size did not match inputs")
                return vectors

        tags = httpx.get(f"{base_url}/api/tags", timeout=30)
        tags.raise_for_status()
        installed = tags.json().get("models", [])
        digest = next((item.get("digest") for item in installed
                       if isinstance(item, dict) and item.get("name") == model), None)
        if not digest:
            raise RuntimeError(f"Ollama embedding model is not installed: {model}")
        options["resolved_ollama_model_digest"] = digest
        return OllamaEmbedder()

    if embedding.startswith("huggingface:"):
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        model_spec = embedding.removeprefix("huggingface:")
        model, separator, revision = model_spec.partition("@")
        cache_dir = options.get("hf_cache")
        local_path = snapshot_download(
            model, revision=revision if separator else None,
            cache_dir=cache_dir, local_files_only=True,
        )
        options["resolved_huggingface_snapshot"] = Path(local_path).name
        embedder = SentenceTransformer(local_path)
        embedder.max_seq_length = 256
        return embedder
    raise ValueError("embedding must begin with 'ollama:' or 'huggingface:'")


class DenseProjectRetriever:
    def __init__(self, config: EvalConfig, ids: list[str], texts: list[str], vectors: Any, embedder: Any) -> None:
        self.config = config
        self.ids = ids
        self.texts = texts
        self.vectors = vectors
        self.embedder = embedder

    def invoke(self, question: str) -> list[Any]:
        import numpy as np

        cache_key = (self.config.embedding, question)
        query = _QUERY_CACHE.get(cache_key)
        if query is None:
            query = self.embedder.encode([
                f"query: {question}" if self.config.embedding.startswith("huggingface:") else question
            ])[0]
            _QUERY_CACHE[cache_key] = query
        query_vector = np.asarray(query, dtype=np.float32)
        matrix = np.asarray(self.vectors, dtype=np.float32)
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
        scores = (matrix @ query_vector) / norms
        best_by_document: dict[str, tuple[float, int]] = {}
        for index, score in enumerate(scores):
            document_id = self.ids[index]
            candidate = (float(score), index)
            if document_id not in best_by_document or candidate[0] > best_by_document[document_id][0]:
                best_by_document[document_id] = candidate
        ranked = sorted(best_by_document.items(), key=lambda item: item[1][0], reverse=True)[:self.config.top_k]
        return [SimpleNamespace(
            page_content=self.texts[index],
            metadata={"document_id": document_id, "score": score},
        ) for document_id, (score, index) in ranked]


def build_retriever(config: EvalConfig, options: dict[str, Any]) -> DenseProjectRetriever:
    """Inject a local embedding client and stable source corpus for dense retrieval."""
    root = Path(options.get("root", ".")).resolve()
    key = (config.embedding, str(root))
    if key not in _INDEX_CACHE:
        document_ids, texts, corpus_fingerprint = _corpus(root)
        options["corpus_fingerprint"] = corpus_fingerprint
        options["corpus_file_count"] = len(document_ids)
        ids, chunks = _chunks(document_ids, texts)
        options["corpus_chunk_count"] = len(chunks)
        embedder = _embedding_client(config.embedding, options)
        passage_texts = [f"passage: {text}" for text in chunks] if config.embedding.startswith("huggingface:") else chunks
        vectors = embedder.encode(passage_texts, batch_size=64) if config.embedding.startswith("huggingface:") else embedder.encode(passage_texts)
        _INDEX_CACHE[key] = (ids, chunks, vectors, embedder)
    ids, texts, vectors, embedder = _INDEX_CACHE[key]
    return DenseProjectRetriever(config, ids, texts, vectors, embedder)
