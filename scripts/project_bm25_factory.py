"""Runnable BM25 baseline factory for the checked-in project smoke set."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.eval.dataset import load_cases
from src.eval.runner import EvalConfig
from src.rag.models import Chunk
from src.rag.retrieval import BM25Retriever


class _EvalRetriever:
    def __init__(self, retriever: BM25Retriever, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def invoke(self, question: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                page_content=result.text,
                metadata={**dict(result.metadata), "score": result.score},
            )
            for result in self._retriever.retrieve(question, k=self._top_k)
        ]


def build_retriever(config: EvalConfig, options: dict[str, Any]) -> _EvalRetriever:
    """Index the source files cited by the smoke set and inject BM25."""
    root = Path(options.get("root", ".")).resolve()
    cases_path = root / options.get("cases", "eval_data/project_source_cases.jsonl")
    cases = load_cases(cases_path)
    source_paths = {span.path for case in cases for span in case.source_spans}
    excluded = {
        ".git", ".venv", "venv", "__pycache__", "node_modules", "mlruns", "eval_data",
        ".cache", ".hf_cache", ".pytest_cache", "uploads", "storage", "data", "build", "dist",
    }
    text_suffixes = {".md", ".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".sh", ".bat"}
    for directory, child_dirs, files in os.walk(root):
        child_dirs[:] = [name for name in child_dirs if name not in excluded and not name.startswith(".")]
        relative_dir = Path(directory).relative_to(root)
        for name in files:
            relative_path = (relative_dir / name).as_posix()
            path = Path(directory) / name
            if (Path(name).suffix.lower() in text_suffixes and not name.startswith(".env")
                    and not name.endswith("lock.json") and path.stat().st_size <= 1_000_000):
                source_paths.add(relative_path)

    chunks = []
    for source_path in sorted(source_paths):
        path = root / source_path
        normalized_source = source_path.replace("\\", "/")
        document_id = f"project:{normalized_source}"
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        chunks.append(Chunk(
            id=document_id,
            text=text,
            metadata={"document_id": document_id, "source": source_path},
        ))
    if not chunks:
        raise ValueError("project smoke cases contain no source documents")
    return _EvalRetriever(BM25Retriever(chunks), config.top_k)
