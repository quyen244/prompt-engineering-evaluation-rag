"""Optional reranking and answer-generation adapters.

The prompt is a separate boundary from the LangChain runnable.  This keeps
prompt construction deterministic and makes the provider-backed runnable
replaceable in tests and in evaluation jobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .models import RetrievedSource


class Runnable(Protocol):
    """Minimal invocation contract shared by LangChain and test runnables."""

    def invoke(self, inputs: Mapping[str, str]) -> Any: ...


@dataclass(frozen=True, slots=True)
class GroundedPrompt:
    """Build the only inputs an answer runnable is allowed to receive."""

    system_message: str = (
        "Answer the question using only the sources below. If the sources do not "
        "contain the answer, say you do not have enough information. Do not invent "
        "citations or facts."
    )

    def inputs(self, question: str, sources: Sequence[RetrievedSource]) -> dict[str, str]:
        context = "\n\n".join(
            f"[source:{source.chunk_id}] {source.text}" for source in sources
        )
        return {
            "question": question,
            "context": context,
            "system_message": self.system_message,
        }


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

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 512,
        *,
        prompt: GroundedPrompt | None = None,
        runnable: Runnable | None = None,
    ) -> None:
        self.prompt = prompt or GroundedPrompt()
        self._runnable = runnable or self._create_runnable(
            model, api_key, base_url, max_tokens
        )

    def _create_runnable(
        self, model: str, api_key: str, base_url: str, max_tokens: int
    ) -> Runnable:
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("answer generation requires langchain-openai") from exc
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_message}"),
                ("human", "Sources:\n{context}\n\nQuestion: {question}"),
            ]
        )
        return prompt | llm | StrOutputParser()

    def generate(self, question: str, sources: Sequence[RetrievedSource]) -> str:
        response = self._runnable.invoke(self.prompt.inputs(question, sources))
        return str(getattr(response, "content", response)).strip()


class ExtractiveGenerator:
    """Deterministic offline generator useful for smoke tests and demos."""

    def generate(self, question: str, sources: Sequence[RetrievedSource]) -> str:
        if not sources:
            return "I do not have enough information to answer that."
        return sources[0].text
