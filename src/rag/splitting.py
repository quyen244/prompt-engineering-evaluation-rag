"""LangChain text splitting with source metadata preservation."""
from __future__ import annotations

import hashlib
from typing import Sequence

from .config import PipelineConfig
from .models import Chunk, SourceDocument


class LangChainSplitter:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError(
                "text splitting requires langchain-text-splitters; install requirements/base.txt"
            ) from exc
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )

    def split(self, documents: Sequence[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            # split_text is intentionally used rather than constructing a new
            # Document, so the loader's source metadata remains authoritative.
            pieces = self._splitter.split_text(document.text)
            for index, text in enumerate(pieces):
                digest = hashlib.sha1(
                    f"{document.metadata.get('source_id', '')}:{index}:{text}".encode()
                ).hexdigest()[:16]
                metadata = dict(document.metadata)
                metadata.update({"chunk_index": index, "chunk_id": digest})
                chunks.append(Chunk(id=digest, text=text, metadata=metadata))
        return chunks


# Kept as a discoverable alias for callers migrating from rag.chunking.
ChunkSplitter = LangChainSplitter
