"""Typed contracts used by the evaluator.

All IDs in these models are content/provenance identifiers, not positional
indexes invented by the evaluator.  This makes a result auditable after a
source dataset changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class EvalMode(str, Enum):
    SMOKE = "smoke"
    IMPROVEMENT = "improvement"
    VALIDATION = "validation"


@dataclass(frozen=True)
class SourceSpan:
    """A citation into a checked-in project source file."""

    path: str
    start_line: int
    end_line: int
    quote: str

    def __post_init__(self) -> None:
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("source span line range is invalid")
        if not self.quote.strip():
            raise ValueError("source span quote cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "quote": self.quote,
        }


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    expected_answer: str
    relevant_document_ids: tuple[str, ...]
    source: str
    source_spans: tuple[SourceSpan, ...] = ()
    dataset_id: str | None = None
    split: str | None = None
    row_id: str | None = None
    row_hash: str | None = None
    answer_spans: tuple[Mapping[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.case_id or not self.question.strip() or not self.expected_answer.strip():
            raise ValueError("case_id, question, and expected_answer are required")
        if not self.relevant_document_ids:
            raise ValueError("at least one relevant document ID is required")
        if self.source == "huggingface" and not all((self.dataset_id, self.split, self.row_id, self.row_hash)):
            raise ValueError("Hugging Face cases require dataset, split, row_id, and row_hash")
        if self.source == "project" and not self.source_spans:
            raise ValueError("project cases require at least one source span")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvalCase":
        spans = tuple(SourceSpan(**span) for span in value.get("source_spans", []))
        return cls(
            case_id=str(value["case_id"]),
            question=str(value["question"]),
            expected_answer=str(value["expected_answer"]),
            relevant_document_ids=tuple(str(x) for x in value["relevant_document_ids"]),
            source=str(value["source"]),
            source_spans=spans,
            dataset_id=value.get("dataset_id"),
            split=value.get("split"),
            row_id=value.get("row_id"),
            row_hash=value.get("row_hash"),
            answer_spans=tuple(dict(span) for span in value.get("answer_spans", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "case_id": self.case_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "relevant_document_ids": list(self.relevant_document_ids),
            "source": self.source,
        }
        if self.source_spans:
            result["source_spans"] = [span.to_dict() for span in self.source_spans]
        for key in ("dataset_id", "split", "row_id", "row_hash"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.answer_spans:
            result["answer_spans"] = [dict(span) for span in self.answer_spans]
        return result


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: str
    rank: int
    score: float | None = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("retrieval ranks start at 1")


@dataclass(frozen=True)
class QueryResult:
    case_id: str
    question: str
    retrieved: tuple[RetrievedDocument, ...]
    metrics: Mapping[str, float]
    answer: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "retrieved": [doc.__dict__ for doc in self.retrieved],
            "metrics": dict(self.metrics),
            "answer": self.answer,
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class EvalReport:
    run_id: str
    mode: EvalMode
    metrics: Mapping[str, float]
    query_results: tuple[QueryResult, ...]
    config: Mapping[str, Any]
    history_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode.value,
            "metrics": dict(self.metrics),
            "config": dict(self.config),
            "query_results": [result.to_dict() for result in self.query_results],
            "history_path": self.history_path,
        }
