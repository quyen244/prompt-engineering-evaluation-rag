"""Compatibility chunker facade backed by LangChain text splitters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import PipelineConfig
from .models import SourceDocument
from .splitting import LangChainSplitter


class ChunkerType:
    SENTENCE = "sentence"
    TOKEN = "token"
    SEMANTIC = "semantic"
    WINDOW = "window"


@dataclass
class _CompatChunker:
    splitter: LangChainSplitter

    def get_nodes_from_documents(self, documents: list[Any]) -> list[Any]:
        normalized = []
        for document in documents:
            if isinstance(document, SourceDocument):
                normalized.append(document)
            else:
                text = getattr(document, "page_content", getattr(document, "text", str(document)))
                metadata = getattr(document, "metadata", {})
                normalized.append(SourceDocument(text=text, metadata=metadata))
        return self.splitter.split(normalized)


class ChunkerFactory:
    @staticmethod
    def create_chunker(chunker_type: str, **kwargs: Any) -> _CompatChunker:
        if chunker_type not in {
            ChunkerType.SENTENCE,
            ChunkerType.TOKEN,
            ChunkerType.SEMANTIC,
            ChunkerType.WINDOW,
        }:
            raise ValueError(f"Unknown chunker type: {chunker_type}")
        config = PipelineConfig(
            chunk_size=int(kwargs.pop("chunk_size", 512)),
            chunk_overlap=int(kwargs.pop("chunk_overlap", 64)),
        )
        return _CompatChunker(LangChainSplitter(config))
