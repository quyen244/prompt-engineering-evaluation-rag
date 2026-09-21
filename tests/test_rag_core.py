from __future__ import annotations

from src.rag.config import PipelineConfig
from src.rag.models import SourceDocument
from src.rag.pipeline import RAGPipeline
from src.rag.retrieval import BM25Retriever


class FakeEmbedder:
    def embed_documents(self, texts):
        return [[float(text.lower().count(word)) for word in ("cats", "dogs")] for text in texts]

    def embed_query(self, text):
        return [float(text.lower().count(word)) for word in ("cats", "dogs")]


class IdentitySplitter:
    def split(self, documents):
        from src.rag.models import Chunk

        return [Chunk(f"chunk-{i}", doc.text, doc.metadata) for i, doc in enumerate(documents)]


class RecordingGenerator:
    def __init__(self):
        self.sources = None

    def generate(self, question, sources):
        self.sources = tuple(sources)
        return "answer from retrieved context"


def test_configuration_isolation():
    first = PipelineConfig(chunk_size=100, chunk_overlap=10)
    second = PipelineConfig(chunk_size=200, chunk_overlap=20)
    assert first != second
    assert first.chunk_size == 100
    assert second.chunk_size == 200
    try:
        first.chunk_size = 300
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("PipelineConfig must be immutable")


def test_retrieval_result_shape_and_real_metadata():
    generator = RecordingGenerator()
    pipeline = RAGPipeline(
        PipelineConfig(retrieval_k=1),
        embedder=FakeEmbedder(),
        generator=generator,
        splitter=IdentitySplitter(),
    )
    pipeline.ingest_documents([
        SourceDocument("cats are independent", {"source": "cats.txt"}),
        SourceDocument("dogs are friendly", {"source": "dogs.txt"}),
    ])
    answer = pipeline.query("cats")
    assert answer.text == "answer from retrieved context"
    assert len(answer.sources) == 1
    source = answer.sources[0]
    assert source.metadata["source"] == "cats.txt"
    assert source.chunk_id == "chunk-0"
    assert source.retriever == "hybrid_rrf"
    assert generator.sources[0].metadata["source"] == "cats.txt"


def test_bm25_does_not_fabricate_sources():
    from src.rag.models import Chunk

    retriever = BM25Retriever([Chunk("known", "only source text", {"source": "real.md"})])
    results = retriever.retrieve("term absent")
    assert len(results) == 1
    assert results[0].text == "only source text"
    assert results[0].metadata["source"] == "real.md"
    assert all(result.chunk_id == "known" for result in results)
