"""Adapter for LangChain retrievers (the retriever implementation stays external)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import RetrievedDocument


def _metadata(document: Any) -> Mapping[str, Any]:
    value = getattr(document, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _document_id(document: Any, index: int) -> str:
    metadata = _metadata(document)
    for key in ("document_id", "doc_id", "id", "source"):
        if metadata.get(key) is not None:
            return str(metadata[key])
    # A retriever without stable IDs cannot be evaluated truthfully.
    raise ValueError(f"retrieved LangChain Document at rank {index} has no stable metadata ID")


def invoke_retriever(retriever: Any, question: str, k: int) -> tuple[RetrievedDocument, ...]:
    """Call a LangChain retriever and normalize Documents without changing it."""
    if hasattr(retriever, "invoke"):
        documents = retriever.invoke(question)
    elif hasattr(retriever, "get_relevant_documents"):
        documents = retriever.get_relevant_documents(question)
    else:
        raise TypeError("retriever must implement LangChain invoke() or get_relevant_documents()")
    if documents is None:
        raise RuntimeError("retriever returned no result; refusing to score missing data")
    result = []
    for rank, document in enumerate(list(documents)[:k], start=1):
        metadata = _metadata(document)
        score = metadata.get("score")
        result.append(RetrievedDocument(
            document_id=_document_id(document, rank), rank=rank,
            score=float(score) if score is not None else None,
            text=getattr(document, "page_content", None), metadata=metadata,
        ))
    return tuple(result)
