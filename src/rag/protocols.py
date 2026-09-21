"""Small protocols that keep providers and evaluation independently testable."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .models import Chunk, RetrievedSource, SourceDocument


@runtime_checkable
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedSource]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, sources: Sequence[RetrievedSource]) -> list[RetrievedSource]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, question: str, sources: Sequence[RetrievedSource]) -> str: ...


@runtime_checkable
class Loader(Protocol):
    def load(self, path: str) -> list[SourceDocument]: ...


@runtime_checkable
class Splitter(Protocol):
    def split(self, documents: Sequence[SourceDocument]) -> list[Chunk]: ...
