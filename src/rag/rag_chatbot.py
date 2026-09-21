"""Backward-compatible facade over the explicit LangChain RAG pipeline."""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from .config import Config, PipelineConfig
from .ingestion import IngestionPipeline
from .models import Answer, SourceDocument
from .pipeline import RAGPipeline, create_openrouter_pipeline

load_dotenv()


class RAGChatBot:
    """Compatibility facade; each instance owns an isolated immutable config."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        model_name: str = Config.MODEL_NAME,
        max_tokens: int = 512,
        *,
        pipeline: RAGPipeline | None = None,
    ) -> None:
        self.config = PipelineConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            generation_model=model_name,
            max_tokens=max_tokens,
        )
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.documents: list[SourceDocument] = []
        self.pipeline = pipeline
        if data_dir is not None:
            self._ensure_pipeline()
            self.reload_index(data_dir)

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")
        self.pipeline = create_openrouter_pipeline(self.config, api_key=api_key)

    def _ensure_index(self) -> RAGPipeline:
        if self.pipeline is None or self.pipeline.artifact is None:
            raise ValueError("index not initialized; provide data_dir or ingest documents first")
        return self.pipeline

    def reload_index(self, data_dir: str) -> None:
        self._ensure_pipeline()
        self.documents = IngestionPipeline().ingest(data_dir)
        assert self.pipeline is not None
        self.pipeline.ingest_documents(self.documents)

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        self._ensure_pipeline()
        self.documents.append(SourceDocument(text=text, metadata=metadata or {}))
        assert self.pipeline is not None
        self.pipeline.ingest_documents(self.documents)

    def query(self, question: str) -> Answer:
        return self._ensure_index().query(question)

    def chat(self, message: str) -> str:
        return self.query(message).text


def get_chatbot(
    data_dir: Optional[str] = None,
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    model_name: str = Config.MODEL_NAME,
    max_tokens: int = 512,
) -> RAGChatBot:
    """Create a fresh chatbot; callers own caching (e.g. Streamlit cache_resource)."""
    return RAGChatBot(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
        max_tokens=max_tokens,
    )
