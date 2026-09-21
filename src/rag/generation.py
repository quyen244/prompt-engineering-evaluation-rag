"""Optional reranking and answer generation adapters."""
from __future__ import annotations

from typing import Sequence

from .models import RetrievedSource


class LocalCrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("reranking requires sentence-transformers") from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, sources: Sequence[RetrievedSource]) -> list[RetrievedSource]:
        if not sources:
            return []
        scores = self._model.predict([(query, source.text) for source in sources])
        ranked = sorted(zip(scores, sources), key=lambda pair: (-float(pair[0]), pair[1].chunk_id))
        return [
            RetrievedSource(
                chunk_id=source.chunk_id,
                text=source.text,
                score=float(score),
                metadata=source.metadata,
                retriever="reranked",
            )
            for score, source in ranked
        ]


class LangChainGenerator:
    """ChatOpenAI-compatible generator for OpenRouter or another endpoint."""

    def __init__(self, model: str, api_key: str, base_url: str, max_tokens: int = 512) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("answer generation requires langchain-openai") from exc
        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    def generate(self, question: str, sources: Sequence[RetrievedSource]) -> str:
        context = "\n\n".join(
            f"[source:{source.chunk_id}] {source.text}" for source in sources
        )
        prompt = (
            "Answer the question using only the sources below. If the sources do not "
            "contain the answer, say you do not have enough information. Do not invent "
            "citations or facts.\n\n"
            f"Sources:\n{context}\n\nQuestion: {question}"
        )
        response = self._llm.invoke(prompt)
        return str(getattr(response, "content", response)).strip()


class ExtractiveGenerator:
    """Deterministic offline generator useful for smoke tests and demos."""

    def generate(self, question: str, sources: Sequence[RetrievedSource]) -> str:
        if not sources:
            return "I do not have enough information to answer that."
        return sources[0].text
