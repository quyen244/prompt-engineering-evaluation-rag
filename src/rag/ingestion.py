"""Real-source ingestion adapters."""
from __future__ import annotations

import hashlib
from pathlib import Path
from .models import SourceDocument

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".html"}


def _source_id(path: Path, page: int | None, text: str) -> str:
    value = f"{path.resolve()}:{page or 0}:{text}".encode("utf-8")
    return hashlib.sha1(value).hexdigest()[:16]


class FileLoader:
    """Load text and PDF files without synthesising content or metadata."""

    def load(self, path: str) -> list[SourceDocument]:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"source file not found: {path}")
        suffix = file_path.suffix.lower()
        if suffix in _TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8")
            return self._documents(file_path, text, None)
        if suffix == ".pdf":
            return self._load_pdf(file_path)
        raise ValueError(f"unsupported source type: {suffix or '<none>'}")

    def load_directory(self, path: str) -> list[SourceDocument]:
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"source directory not found: {path}")
        documents: list[SourceDocument] = []
        for file_path in sorted(p for p in directory.rglob("*") if p.is_file()):
            if file_path.suffix.lower() in _TEXT_EXTENSIONS or file_path.suffix.lower() == ".pdf":
                documents.extend(self.load(str(file_path)))
        return documents

    @staticmethod
    def _documents(path: Path, text: str, page: int | None) -> list[SourceDocument]:
        if not text.strip():
            return []
        metadata = {
            "source": str(path.resolve()),
            "file_name": path.name,
            "source_id": _source_id(path, page, text),
        }
        if page is not None:
            metadata["page"] = page
        return [SourceDocument(text=text, metadata=metadata)]

    def _load_pdf(self, path: Path) -> list[SourceDocument]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pypdf") from exc
        documents: list[SourceDocument] = []
        for page_number, page in enumerate(PdfReader(str(path)).pages, 1):
            text = page.extract_text() or ""
            documents.extend(self._documents(path, text, page_number))
        return documents


class IngestionPipeline:
    """Convenience entry point for files and directories."""

    def __init__(self, loader: FileLoader | None = None) -> None:
        self.loader = loader or FileLoader()

    def ingest(self, path: str) -> list[SourceDocument]:
        candidate = Path(path)
        return self.loader.load_directory(path) if candidate.is_dir() else self.loader.load(path)
