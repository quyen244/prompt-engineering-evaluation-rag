from __future__ import annotations

from src.rag.config import PipelineConfig
from src.rag.evaluation import EvaluationRecord
from src.rag.generation import GroundedPrompt, LangChainGenerator
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


class RecordingRunnable:
    def __init__(self):
        self.inputs = None

    def invoke(self, inputs):
        self.inputs = inputs
        return "runnable answer"


class FixedRetriever:
    def retrieve(self, query, *, k=None):
        from src.rag.models import RetrievedSource

        return [
            RetrievedSource(
                "persisted", "persisted source", 1.0, {"source": "qdrant"}
            )
        ]


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


def test_prompt_and_runnable_are_independent_boundaries():
    runnable = RecordingRunnable()
    generator = LangChainGenerator(
        "test-model",
        "test-key",
        "https://example.invalid/v1",
        prompt=GroundedPrompt(),
        runnable=runnable,
    )
    from src.rag.models import RetrievedSource

    source = [RetrievedSource("c1", "known context", 1.0, {"source": "real.md"})]
    assert generator.generate("What?", source) == "runnable answer"
    assert runnable.inputs == {
        "question": "What?",
        "context": "[source:c1] known context",
        "system_message": GroundedPrompt().system_message,
    }


def test_injected_retriever_remains_the_retriever_boundary():
    generator = RecordingGenerator()
    pipeline = RAGPipeline(
        PipelineConfig(retrieval_k=1),
        embedder=FakeEmbedder(),
        generator=generator,
        retriever=FixedRetriever(),
        splitter=IdentitySplitter(),
    )
    pipeline.ingest_documents([SourceDocument("local copy", {"source": "local.md"})])
    assert pipeline.query("anything").sources[0].metadata["source"] == "qdrant"


def test_evaluation_record_preserves_answer_sources():
    from src.rag.models import Answer, RetrievedSource

    record = EvaluationRecord.from_answer(
        "Where?",
        Answer(
            "there",
            (RetrievedSource("c1", "source text", 0.9, {"source": "real.md"}, "dense"),),
        ),
    )
    assert record.as_mapping()["sources"][0]["metadata"] == {"source": "real.md"}
