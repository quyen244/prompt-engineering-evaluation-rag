"""Dense, BM25, and hybrid retrieval adapters with one result contract."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .models import Chunk, RetrievedSource
from .protocols import Embedder, Retriever

_TOKEN = re.compile(r"(?u)\b\w+\b")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _source(chunk: Chunk, score: float, retriever: str) -> RetrievedSource:
    return RetrievedSource(
        chunk_id=chunk.id,
        text=chunk.text,
        score=float(score),
        metadata=chunk.metadata,
        retriever=retriever,
    )


class DenseRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks = tuple(chunks)
        self.embedder = embedder
        self._vectors = tuple(embedder.embed_documents([chunk.text for chunk in self.chunks]))

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedSource]:
        if not query.strip() or not self.chunks:
            return []
        query_vector = self.embedder.embed_query(query)
        scored = [(_cosine(query_vector, vector), chunk) for chunk, vector in zip(self.chunks, self._vectors)]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [_source(chunk, score, "dense") for score, chunk in scored[: k or len(scored)]]


class BM25Retriever:
    """Small dependency-free BM25 implementation with deterministic ranking."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = tuple(chunks)
        self.k1, self.b = k1, b
        self._terms = tuple(_tokens(chunk.text) for chunk in self.chunks)
        self._lengths = tuple(len(tokens) for tokens in self._terms)
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 1.0
        self._document_frequency: Counter[str] = Counter()
        for terms in self._terms:
            self._document_frequency.update(set(terms))

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedSource]:
        if not query.strip() or not self.chunks:
            return []
        query_terms = _tokens(query)
        document_count = len(self.chunks)
        scores: list[tuple[float, Chunk]] = []
        for chunk, terms, length in zip(self.chunks, self._terms, self._lengths):
            counts = Counter(terms)
            score = 0.0
            for term in query_terms:
                df = self._document_frequency.get(term, 0)
                if not df:
                    continue
                idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                frequency = counts[term]
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self._avg_length)
                score += idf * frequency * (self.k1 + 1) / denominator if denominator else 0.0
            scores.append((score, chunk))
        scores.sort(key=lambda item: (-item[0], item[1].id))
        return [_source(chunk, score, "bm25") for score, chunk in scores[: k or len(scores)]]


class RRFHybridRetriever:
    """Reciprocal Rank Fusion over any retriever implementations."""

    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        self.retrievers = tuple(retrievers)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedSource]:
        fused: dict[str, tuple[float, RetrievedSource]] = {}
        for retriever in self.retrievers:
            for rank, source in enumerate(retriever.retrieve(query, k=k), 1):
                score, previous = fused.get(source.chunk_id, (0.0, source))
                fused[source.chunk_id] = (score + 1.0 / (self.rrf_k + rank), previous)
        ranked = sorted(fused.values(), key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievedSource(
                chunk_id=source.chunk_id,
                text=source.text,
                score=score,
                metadata=source.metadata,
                retriever="hybrid_rrf",
            )
            for score, source in ranked[: k or len(ranked)]
        ]


class QdrantRetriever:
    """Adapter boundary for a persisted Qdrant collection.

    Construction is lazy so core unit tests do not need a running Qdrant service.
    """

    def __init__(self, client: object, collection_name: str, embedder: Embedder) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievedSource]:
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=self.embedder.embed_query(query),
            limit=k or 5,
        )
        return [
            RetrievedSource(
                chunk_id=str(point.id),
                text=str(point.payload.get("text", "")),
                score=float(point.score),
                metadata=point.payload.get("metadata", {}),
                retriever="qdrant",
            )
            for point in response
            if point.payload.get("text")
        ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
