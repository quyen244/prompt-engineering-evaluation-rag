"""Per-pipeline immutable configuration.

No module-level mutable model, retriever, or chatbot state is kept here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_k: int = 5
    dense_k: int = 10
    sparse_k: int = 10
    rrf_k: int = 60
    rerank_k: int | None = None
    generation_model: str = "inclusionai/ling-3.0-flash"
    max_tokens: int = 512
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        for name in ("retrieval_k", "dense_k", "sparse_k", "rrf_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rerank_k is not None and self.rerank_k <= 0:
            raise ValueError("rerank_k must be positive when provided")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


# Compatibility for callers that used Config.MODEL_NAME. Unlike the old class,
# this value is not used to hold mutable provider instances.
class Config:
    MODEL_NAME = PipelineConfig.generation_model
