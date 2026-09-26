"""Evaluate the lightweight TechQA corpus with BM25, dense, and hybrid retrieval."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from src.eval.metrics.retrieval import (
    hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k,
)
from src.rag.models import Chunk
from src.rag.retrieval import BM25Retriever, DenseRetriever, RRFHybridRetriever

DEFAULT_CORPUS = "eval_data/corpus.jsonl"
DEFAULT_QUESTIONS = "eval_data/eval_questions.json"
DEFAULT_EMBEDDING = "ollama:qwen3-embedding:0.6b"
VERSION = "1.0.0"
CHUNK_SIZE = 850
CHUNK_OVERLAP = 150


def load_dataset(
    corpus_path: str | Path = DEFAULT_CORPUS,
    questions_path: str | Path = DEFAULT_QUESTIONS,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Load and validate both inputs; fingerprint their exact bytes in corpus/question order."""
    corpus_bytes = Path(corpus_path).read_bytes()
    question_bytes = Path(questions_path).read_bytes()
    documents = [json.loads(line) for line in corpus_bytes.decode("utf-8").splitlines() if line.strip()]
    questions = json.loads(question_bytes)
    if not isinstance(questions, list) or not documents or not questions:
        raise ValueError("corpus and questions must be nonempty lists")
    doc_ids: set[str] = set()
    for document in documents:
        if not isinstance(document, dict) or any(not isinstance(document.get(field), str) or not document[field].strip()
                                                     for field in ("doc_id", "title", "text")):
            raise ValueError("each corpus row needs nonempty doc_id, title, and text")
        if document["doc_id"] in doc_ids:
            raise ValueError(f"duplicate corpus doc_id: {document['doc_id']}")
        doc_ids.add(document["doc_id"])
    question_ids: set[str] = set()
    for question in questions:
        if not isinstance(question, dict) or any(not isinstance(question.get(field), str) or not question[field].strip()
                                                 for field in ("id", "question", "ground_truth", "gold_doc_id")):
            raise ValueError("each question needs nonempty id, question, ground_truth, and gold_doc_id")
        if question["id"] in question_ids:
            raise ValueError(f"duplicate question id: {question['id']}")
        question_ids.add(question["id"])
        if question["gold_doc_id"] not in doc_ids:
            raise ValueError(f"gold_doc_id absent from corpus: {question['gold_doc_id']}")
    digest = hashlib.sha256()
    for label, data in (("corpus", corpus_bytes), ("questions", question_bytes)):
        digest.update(label.encode("ascii") + b"\0" + str(len(data)).encode("ascii") + b"\0" + data)
    return documents, questions, digest.hexdigest()


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        import httpx

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=120)
        self._document_vectors: dict[tuple[str, ...], list[list[float]]] = {}
        response = self.client.get(f"{self.base_url}/api/tags", timeout=15)
        response.raise_for_status()
        installed = response.json().get("models", [])
        match = next((row for row in installed if row.get("name") == model), None)
        if match is None:
            raise RuntimeError(f"Ollama embedding model is not installed: {model}")
        self.model_digest = str(match.get("digest", "unknown"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        key = tuple(texts) if len(texts) > 1 else None
        if key is not None and key in self._document_vectors:
            return self._document_vectors[key]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            response = self.client.post(f"{self.base_url}/api/embed", json={
                "model": self.model, "input": batch, "truncate": True,
            })
            response.raise_for_status()
            received = response.json().get("embeddings", [])
            if len(received) != len(batch):
                raise RuntimeError("Ollama embedding response count does not match batch")
            vectors.extend(received)
            if len(texts) > 32:
                print(f"embedding passages: {len(vectors)}/{len(texts)}", file=sys.stderr)
        if len(vectors) != len(texts):
            raise RuntimeError("Ollama embedding response count does not match corpus")
        if key is not None:
            self._document_vectors[key] = vectors
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalHFEmbedder:
    """Load an already cached E5 model without downloading anything."""

    def __init__(self, model: str) -> None:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        model_name, separator, revision = model.partition("@")
        snapshot = snapshot_download(model_name, revision=revision if separator else None,
                                     local_files_only=True)
        self.model_digest = Path(snapshot).name
        self.model = SentenceTransformer(snapshot)
        self._document_vectors: dict[tuple[str, ...], list[list[float]]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        key = tuple(texts)
        if key not in self._document_vectors:
            self._document_vectors[key] = self.model.encode(
                [f"passage: {text}" for text in texts], batch_size=32,
                normalize_embeddings=True,
            ).tolist()
        return self._document_vectors[key]

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()


class TechQARetriever:
    def __init__(self, documents: list[dict[str, str]], method: str, embedder: Any = None) -> None:
        self.documents = {doc["doc_id"]: doc for doc in documents}
        chunks: list[Chunk] = []
        for doc in documents:
            body = doc["text"]
            step = CHUNK_SIZE - CHUNK_OVERLAP
            for index, start in enumerate(range(0, len(body), step)):
                passage = body[start:start + CHUNK_SIZE]
                if not passage.strip():
                    continue
                chunks.append(Chunk(
                    id=f"{doc['doc_id']}::chunk-{index:04d}",
                    text=f"{doc['title']}\n\n{passage}",
                    metadata={"document_id": doc["doc_id"], "chunk_index": index},
                ))
                if start + CHUNK_SIZE >= len(body):
                    break
        self.chunk_count = len(chunks)
        if method == "bm25":
            self.backend = BM25Retriever(chunks)
        elif method == "dense":
            if embedder is None:
                raise ValueError("dense retrieval requires an embedder")
            self.backend = DenseRetriever(chunks, embedder)
        elif method == "hybrid":
            if embedder is None:
                raise ValueError("hybrid retrieval requires an embedder")
            self.backend = RRFHybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, embedder)])
        else:
            raise ValueError(f"unknown retrieval method: {method}")
        self.method = method

    def retrieve(self, question: str, k: int) -> list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be >= 1")
        # Rank all passages, then keep the highest ranked passage per document.
        # Full candidate lists also make RRF scores independent of output k.
        sources = self.backend.retrieve(question, k=self.chunk_count)
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in sources:
            doc_id = str(item.metadata["document_id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append({**self.documents[doc_id], "score": item.score,
                            "chunk_id": item.chunk_id, "context_text": item.text})
            if len(results) == k:
                break
        return results


def build_retriever(
    documents: list[dict[str, str]], *, method: str = "dense",
    embedding: str = DEFAULT_EMBEDDING,
    ollama_base_url: str = "http://localhost:11434",
    embedder: Any = None,
) -> TechQARetriever:
    """Build an in-memory retriever. An injected embedder supports offline tests."""
    if method not in {"bm25", "dense", "hybrid"}:
        raise ValueError("method must be bm25, dense, or hybrid")
    if method != "bm25" and embedder is None:
        if embedding.startswith("ollama:"):
            embedder = OllamaEmbedder(embedding.removeprefix("ollama:"), ollama_base_url)
        elif embedding.startswith("huggingface:"):
            embedder = LocalHFEmbedder(embedding.removeprefix("huggingface:"))
        else:
            raise ValueError("embedding must use ollama:<installed-model> or huggingface:<cached-model>")
    return TechQARetriever(documents, method, embedder)


def evaluate_retriever(
    retriever: TechQARetriever, questions: list[dict[str, str]], top_k: int,
    ranked_results: list[list[dict[str, Any]]] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metric_functions = {
        "hit_rate": hit_rate_at_k, "precision": precision_at_k,
        "recall": recall_at_k, "mrr": mrr_at_k, "ndcg": ndcg_at_k,
    }
    rows: list[dict[str, Any]] = []
    if ranked_results is not None and len(ranked_results) != len(questions):
        raise ValueError("ranked result count does not match question count")
    for index, question in enumerate(questions):
        retrieved = (ranked_results[index][:top_k] if ranked_results is not None
                     else retriever.retrieve(question["question"], top_k))
        ids = [item["doc_id"] for item in retrieved]
        metrics = {name: function(ids, [question["gold_doc_id"]], top_k)
                   for name, function in metric_functions.items()}
        rows.append({"question_id": question["id"], "gold_doc_id": question["gold_doc_id"],
                     "retrieved_doc_ids": ids, "retrieved_chunk_ids": [item["chunk_id"] for item in retrieved],
                     "scores": [item["score"] for item in retrieved],
                     "metrics": metrics})
    means = {f"{name}_at_{top_k}": sum(row["metrics"][name] for row in rows) / len(rows)
             for name in metric_functions}
    return means, rows


def run_ablation(
    documents: list[dict[str, str]], questions: list[dict[str, str]], dataset_fingerprint: str,
    *, methods: list[str], top_ks: list[int], embedding: str = DEFAULT_EMBEDDING,
    embeddings: list[str] | None = None,
    corpus_sha256: str = "unknown", questions_sha256: str = "unknown",
    ollama_base_url: str = "http://localhost:11434", mlflow_module: Any = None,
    experiment: str = "techqa-retrieval", tracking_uri: str = "sqlite:///mlflow.db",
    champion_path: str | Path = "eval_data/results/techqa_retrieval_champion.json",
) -> list[dict[str, Any]]:
    if not methods or len(set(methods)) != len(methods) or any(method not in {"bm25", "dense", "hybrid"} for method in methods):
        raise ValueError("methods must be unique selections from bm25, dense, hybrid")
    if not top_ks or len(set(top_ks)) != len(top_ks) or any(k < 1 for k in top_ks):
        raise ValueError("top_ks must be unique positive integers")
    embedding_values = embeddings if embeddings is not None else [embedding]
    if not embedding_values or len(set(embedding_values)) != len(embedding_values):
        raise ValueError("embeddings must be a nonempty list of unique model IDs")
    if mlflow_module is None:
        import mlflow as mlflow_module
    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment(experiment)
    study_id = uuid.uuid4().hex
    results = []
    embedders: dict[str, Any] = {}
    for method in methods:
        for model in (["none"] if method == "bm25" else embedding_values):
            if method != "bm25" and model not in embedders:
                if model.startswith("ollama:"):
                    embedders[model] = OllamaEmbedder(model.removeprefix("ollama:"), ollama_base_url)
                elif model.startswith("huggingface:"):
                    embedders[model] = LocalHFEmbedder(model.removeprefix("huggingface:"))
                else:
                    raise ValueError("embedding must use ollama: or huggingface: prefix")
            retriever = build_retriever(documents, method=method, embedding=model,
                                        ollama_base_url=ollama_base_url,
                                        embedder=embedders.get(model))
            digest = getattr(getattr(retriever.backend, "embedder", None), "model_digest", "none")
            # Hybrid owns both backends, so derive the digest from its dense member.
            if method == "hybrid":
                digest = getattr(retriever.backend.retrievers[1].embedder, "model_digest", "unknown")
            ranked_results = [retriever.retrieve(question["question"], max(top_ks))
                              for question in questions]
            for k in top_ks:
                with mlflow_module.start_run(run_name=f"{method}-{model.replace(':', '-')}-topk-{k}") as run:
                    mlflow_module.set_tags({"run_kind": "techqa_retrieval_ablation",
                                            "dataset_fingerprint": dataset_fingerprint,
                                            "study_id": study_id})
                    mlflow_module.log_params({"method": method, "top_k": k, "embedding": model,
                                               "embedding_digest": digest, "corpus_size": len(documents),
                                               "question_count": len(questions), "dataset_fingerprint": dataset_fingerprint,
                                               "corpus_sha256": corpus_sha256, "questions_sha256": questions_sha256,
                                               "chunk_size_chars": CHUNK_SIZE, "chunk_overlap_chars": CHUNK_OVERLAP,
                                               "chunk_count": retriever.chunk_count,
                                               "document_rank_policy": "best_ranked_chunk_per_doc"})
                    metrics, rows = evaluate_retriever(retriever, questions, k, ranked_results)
                    mlflow_module.log_metrics(metrics)
                    mlflow_module.log_dict({"study_id": study_id, "dataset_fingerprint": dataset_fingerprint,
                                            "corpus_sha256": corpus_sha256, "questions_sha256": questions_sha256,
                                            "chunk_size_chars": CHUNK_SIZE, "chunk_overlap_chars": CHUNK_OVERLAP,
                                            "chunk_count": retriever.chunk_count,
                                            "document_rank_policy": "best_ranked_chunk_per_doc", "method": method,
                                            "top_k": k, "embedding": model,
                                            "embedding_digest": digest, "metrics": metrics}, "evaluation/summary.json")
                    mlflow_module.log_dict({"results": rows}, "evaluation/per_case.json")
                    results.append({"run_id": run.info.run_id, "method": method, "top_k": k,
                                    "embedding": model, "embedding_digest": digest, "metrics": metrics})
    champion = max(results, key=lambda result: (
        result["metrics"][f"ndcg_at_{result['top_k']}"],
        result["metrics"][f"hit_rate_at_{result['top_k']}"],
        result["metrics"][f"mrr_at_{result['top_k']}"],
        -result["top_k"],
    ))
    artifact = {"study_id": study_id, "method": champion["method"], "embedding": champion["embedding"],
                "top_k": champion["top_k"], "dataset_fingerprint": dataset_fingerprint,
                "corpus_sha256": corpus_sha256, "questions_sha256": questions_sha256,
                "chunk_size_chars": CHUNK_SIZE, "chunk_overlap_chars": CHUNK_OVERLAP,
                "document_rank_policy": "best_ranked_chunk_per_doc",
                "embedding_digest": champion["embedding_digest"],
                "retrieval_run_id": champion["run_id"], "metrics": champion["metrics"]}
    path = Path(champion_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with mlflow_module.start_run(run_id=champion["run_id"]):
        mlflow_module.log_dict(artifact, "comparison/champion.json")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--methods", nargs="+", choices=("bm25", "dense", "hybrid"), default=["bm25", "dense", "hybrid"])
    parser.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING)
    parser.add_argument("--embeddings", nargs="+", help="Evaluate several local embedding models in one paired matrix")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--experiment", default="techqa-retrieval")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--champion-path", default="eval_data/results/techqa_retrieval_champion.json")
    if argv is None:
        import sys
        argv = sys.argv[1:]
    if len(argv) == 1 and argv[0] in {"-v", "-V", "--version"}:
        print(VERSION)
        return 0
    args = parser.parse_args(argv)
    try:
        documents, questions, fingerprint = load_dataset(args.corpus, args.questions)
        if not argv:
            print(f"dataset: {len(documents)} documents, {len(questions)} questions")
            print(f"fingerprint: {fingerprint}")
            print("help[1]: Run `python -m scripts.techqa_retrieval --help` for evaluation options")
            return 0
        results = run_ablation(documents, questions, fingerprint, methods=args.methods, top_ks=args.top_k,
                               embedding=args.embedding, embeddings=args.embeddings,
                               corpus_sha256=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
                               questions_sha256=hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
                               ollama_base_url=args.ollama_base_url,
                               experiment=args.experiment, tracking_uri=args.tracking_uri,
                               champion_path=args.champion_path)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}")
        return 1
    print(f"runs[{len(results)}]{{run_id,method,top_k,hit_rate,mrr,ndcg}}:")
    for result in results:
        scores = result["metrics"]
        k = result["top_k"]
        print(f"  {result['run_id']},{result['method']},{k},{scores[f'hit_rate_at_{k}']:.4f},{scores[f'mrr_at_{k}']:.4f},{scores[f'ndcg_at_{k}']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
