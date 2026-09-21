"""Composable LangChain RAG foundation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import PipelineConfig
from .generation import LangChainGenerator
from .ingestion import IngestionPipeline
from .models import Answer, Chunk, RetrievedSource, SourceDocument
from .protocols import Embedder, Generator, Reranker, Retriever
from .retrieval import BM25Retriever, DenseRetriever, RRFHybridRetriever
from .splitting import LangChainSplitter


@dataclass(frozen=True, slots=True)
class IngestionArtifact:
    documents: tuple[SourceDocument, ...]
    chunks: tuple[Chunk, ...]


class RAGPipeline:
    """Explicit ingestion -> retrieval -> optional reranking -> generation pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        embedder: Embedder,
        generator: Generator,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        splitter: object | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker
        self.splitter = splitter or LangChainSplitter(config)
        self.retriever = retriever
        self._artifact: IngestionArtifact | None = None

    @property
    def artifact(self) -> IngestionArtifact | None:
        return self._artifact

    def ingest_documents(self, documents: Sequence[SourceDocument]) -> IngestionArtifact:
        chunks = tuple(self.splitter.split(documents))
        if not chunks:
            raise ValueError("ingestion produced no non-empty chunks")
        dense = DenseRetriever(chunks, self.embedder)
        sparse = BM25Retriever(chunks)
        self.retriever = RRFHybridRetriever((dense, sparse), self.config.rrf_k)
        self._artifact = IngestionArtifact(tuple(documents), chunks)
        return self._artifact

    def ingest_path(self, path: str) -> IngestionArtifact:
        return self.ingest_documents(IngestionPipeline().ingest(path))

    def retrieve(self, question: str, *, k: int | None = None) -> list[RetrievedSource]:
        if self.retriever is None:
            raise RuntimeError("pipeline has no index; call ingest_documents or ingest_path first")
        sources = self.retriever.retrieve(question, k=k or self.config.dense_k)
        if self.reranker:
            sources = self.reranker.rerank(question, sources)
        return sources[: k or self.config.retrieval_k]

    def query(self, question: str, *, k: int | None = None) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        sources = self.retrieve(question, k=k)
        return Answer(text=self.generator.generate(question, sources), sources=tuple(sources))


def create_openrouter_pipeline(
    config: PipelineConfig,
    *,
    api_key: str,
    reranker: Reranker | None = None,
) -> RAGPipeline:
    """Factory kept separate from the pipeline to make provider swapping explicit."""
    from .embeddings import create_embedder

    generator = LangChainGenerator(
        model=config.generation_model,
        api_key=api_key,
        base_url=config.openrouter_base_url,
        max_tokens=config.max_tokens,
    )
    return RAGPipeline(
        config,
        embedder=create_embedder(config),
        generator=generator,
        reranker=reranker,
    )
