"""Immutable, serialisable data contracts for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _metadata(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(values or {}))


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A real source loaded from a user-provided path or stream."""

    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("source document text cannot be empty")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("chunk text cannot be empty")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class RetrievedSource:
    """One ranked source returned to callers (never an invented citation)."""

    chunk_id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retriever: str = "unknown"

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("retrieved source text cannot be empty")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[RetrievedSource, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("answer text cannot be empty")
