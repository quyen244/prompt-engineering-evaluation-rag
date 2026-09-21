"""Small offline playground for the LangChain RAG foundation.

Run this with a local embedding dependency installed; it deliberately uses no
network endpoint for embeddings.
"""
from __future__ import annotations

from .config import PipelineConfig
from .embeddings import create_embedder
from .models import SourceDocument
from .pipeline import RAGPipeline
from .generation import ExtractiveGenerator


def build_demo() -> RAGPipeline:
    config = PipelineConfig(chunk_size=200, chunk_overlap=20)
    pipeline = RAGPipeline(
        config,
        embedder=create_embedder(config),
        generator=ExtractiveGenerator(),
    )
    pipeline.ingest_documents([
        SourceDocument(
            "Local embeddings keep indexing offline and reproducible.",
            {"source": "playground"},
        )
    ])
    return pipeline
