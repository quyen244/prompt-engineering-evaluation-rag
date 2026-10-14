# LangChain RAG core

For the PDF API and TechQA chunking paths, see [Chunking E2E](chunking_e2e.md).
The API does not call the splitter described in this document.

`src/rag/pipeline.py` is the framework boundary for the real-source RAG path:

1. `IngestionPipeline` loads text/Markdown/CSV/HTML or PDF pages and preserves the
   actual path and page in metadata.
2. `LangChainSplitter` creates chunks without losing that metadata.
3. `LocalHuggingFaceEmbeddings` uses a local sentence-transformers model; the
   OpenRouter key is used only for answer generation.
4. `DenseRetriever` and `BM25Retriever` expose the same `RetrievedSource` shape.
   `RRFHybridRetriever` combines their ranked lists. `QdrantRetriever` is the
   persistence adapter boundary and is lazy to keep unit tests offline.
5. `LocalCrossEncoderReranker` is optional. A generator receives only retrieved
   sources and cannot fabricate source metadata.
6. `GroundedPrompt` and the LangChain runnable are separate injectable seams.
   `EvaluationRecord` and `Evaluator` provide a source-aware evaluation boundary;
   metric implementations remain outside the core and no evaluation results are
   implied by this package.

```python
from src.rag.config import PipelineConfig
from src.rag.models import SourceDocument
from src.rag.pipeline import RAGPipeline

config = PipelineConfig()
pipeline = RAGPipeline(config, embedder=my_embedder, generator=my_generator)
pipeline.ingest_documents([SourceDocument("text from a real file", {"source": "file.txt"})])
answer = pipeline.query("What does the file say?")
for source in answer.sources:
    print(source.metadata["source"], source.score)
```

Injecting `embedder`, `generator`, `retriever`, or `reranker` makes retrieval and
answer generation independently testable. Injecting a retriever preserves that
retriever when documents are ingested, so persisted/vector-store adapters can own
their indexing lifecycle. Use `create_openrouter_pipeline` only at an application
boundary. Do not add a global chatbot singleton; Streamlit owns its own
`cache_resource` lifecycle.
