"""Embedding providers. The default is local Hugging Face, never OpenRouter."""
from __future__ import annotations

from .config import PipelineConfig


class LocalHuggingFaceEmbeddings:
    """Thin LangChain adapter around a local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "local embeddings require langchain-huggingface and sentence-transformers"
            ) from exc
        self._model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def create_embedder(config: PipelineConfig) -> LocalHuggingFaceEmbeddings:
    return LocalHuggingFaceEmbeddings(config.embedding_model)
