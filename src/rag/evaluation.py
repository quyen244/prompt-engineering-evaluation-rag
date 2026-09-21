"""Evaluation boundary for answers produced by the RAG core.

This module does not choose metrics or emit observability data.  It only gives
evaluation jobs a stable, source-aware record and an injectable evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import Answer
from .protocols import Evaluator


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    question: str
    answer: str
    sources: tuple[Mapping[str, object], ...]

    @classmethod
    def from_answer(cls, question: str, answer: Answer) -> "EvaluationRecord":
        return cls(
            question=question,
            answer=answer.text,
            sources=tuple(
                {
                    "chunk_id": source.chunk_id,
                    "text": source.text,
                    "score": source.score,
                    "metadata": dict(source.metadata),
                    "retriever": source.retriever,
                }
                for source in answer.sources
            ),
        )

    def as_mapping(self) -> dict[str, object]:
        """Return a plain mapping suitable for an evaluator or dataset row."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [dict(source) for source in self.sources],
        }


def evaluate_answer(
    evaluator: Evaluator, question: str, answer: Answer
) -> Mapping[str, object]:
    """Delegate metric decisions to an injected evaluator implementation."""
    return evaluator.evaluate(question, answer)
