"""Tune TechQA RRF on 50 questions, then compare three methods on 20 held-out questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from scripts.techqa_retrieval import CHUNK_OVERLAP, CHUNK_SIZE, OllamaEmbedder, build_retriever, load_dataset
from src.eval.metrics.retrieval import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.rag.retrieval import RRFHybridRetriever


VERSION = "1.0.0"
RRF_K_VALUES = (1, 5, 10, 20, 40, 60, 100)
TOP_K_VALUES = (1, 3, 5)
METRICS = {
    "hit_rate": hit_rate_at_k,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class _FrozenRetriever:
    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources

    def retrieve(self, _query: str, *, k: int | None = None) -> list[Any]:
        return self.sources[:k] if k is not None else self.sources


def _document_ranking(sources: list[Any]) -> list[dict[str, Any]]:
    ranked = []
    seen = set()
    for source in sources:
        doc_id = str(source.metadata["document_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append({"doc_id": doc_id, "chunk_id": source.chunk_id, "score": source.score})
    return ranked


def _evaluate(
    questions: list[dict[str, str]], rankings: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = []
    for question in questions:
        ranked = rankings[question["id"]]
        ids = [item["doc_id"] for item in ranked]
        scores = {
            f"{name}_at_{k}": metric(ids, [question["gold_doc_id"]], k)
            for k in TOP_K_VALUES for name, metric in METRICS.items()
        }
        rows.append({
            "question_id": question["id"],
            "gold_doc_id": question["gold_doc_id"],
            "retrieved_doc_ids_at_5": ids[:5],
            "retrieved_chunk_ids_at_5": [item["chunk_id"] for item in ranked[:5]],
            "scores": scores,
        })
    means = {key: sum(row["scores"][key] for row in rows) / len(rows)
             for key in rows[0]["scores"]}
    return means, rows


def _fused_rankings(
    questions: list[dict[str, str]], frozen: dict[str, dict[str, list[Any]]],
    rrf_k: int, chunk_count: int,
) -> dict[str, list[dict[str, Any]]]:
    rankings = {}
    for question in questions:
        item = frozen[question["id"]]
        fusion = RRFHybridRetriever(
            (_FrozenRetriever(item["bm25"]), _FrozenRetriever(item["dense"])),
            rrf_k=rrf_k,
        )
        rankings[question["id"]] = _document_ranking(
            fusion.retrieve(question["question"], k=chunk_count)
        )
    return rankings


def run(study_dir: Path, model: str, ollama_url: str, tracking_uri: str) -> dict[str, Any]:
    import mlflow

    corpus = study_dir / "corpus.jsonl"
    tune_file = study_dir / "tune_questions.json"
    test_file = study_dir / "test_questions.json"
    manifest_file = study_dir / "manifest.json"
    for path in (corpus, tune_file, test_file, manifest_file):
        if not path.is_file():
            raise ValueError(f"study file missing: {path}; run scripts.prepare_techqa_rrf_study first")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for name, path in (("corpus", corpus), ("tune_questions", tune_file), ("test_questions", test_file)):
        if manifest.get(f"{name}_sha256") != _sha256(path):
            raise ValueError(f"study fingerprint mismatch: {name}")
    documents, tune_questions, _ = load_dataset(corpus, tune_file)
    _, test_questions, _ = load_dataset(corpus, test_file)
    if len(tune_questions) != 50 or len(test_questions) != 20:
        raise ValueError("study must have exactly 50 tune and 20 test questions")
    if {row["id"] for row in tune_questions} & {row["id"] for row in test_questions}:
        raise ValueError("tune and test question IDs overlap")
    print(f"embedding: {len(documents)} documents with {model}", file=sys.stderr)
    embedder = OllamaEmbedder(model, ollama_url)
    dense = build_retriever(documents, method="dense", embedder=embedder)
    bm25 = build_retriever(documents, method="bm25")
    if dense.chunk_count != bm25.chunk_count:
        raise RuntimeError("dense and BM25 indexes have different chunk counts")
    questions = tune_questions + test_questions
    frozen = {}
    for index, question in enumerate(questions, 1):
        frozen[question["id"]] = {
            "dense": dense.backend.retrieve(question["question"], k=dense.chunk_count),
            "bm25": bm25.backend.retrieve(question["question"], k=bm25.chunk_count),
        }
        if index % 10 == 0 or index == len(questions):
            print(f"ranked_questions: {index}/{len(questions)}", file=sys.stderr)

    dataset_fingerprint = hashlib.sha256(
        (manifest["corpus_sha256"] + manifest["tune_questions_sha256"]
         + manifest["test_questions_sha256"]).encode("ascii")
    ).hexdigest()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("techqa-rrf-tuning")
    result: dict[str, Any] = {}
    with mlflow.start_run(run_name="rrf-50-tune-20-test") as parent:
        mlflow.log_params({
            "dataset_fingerprint": dataset_fingerprint,
            "corpus_sha256": manifest["corpus_sha256"],
            "tune_questions_sha256": manifest["tune_questions_sha256"],
            "test_questions_sha256": manifest["test_questions_sha256"],
            "source_train_sha256": manifest["train_source_sha256"],
            "source_light_corpus_sha256": manifest["light_corpus_sha256"],
            "source_full_corpus_bytes": manifest["full_corpus_bytes"],
            "selection_seed": manifest["seed"],
            "document_count": len(documents),
            "chunk_count": dense.chunk_count,
            "chunk_size_chars": CHUNK_SIZE,
            "chunk_overlap_chars": CHUNK_OVERLAP,
            "embedding_model": model,
            "embedding_digest": embedder.model_digest,
            "tune_question_count": len(tune_questions),
            "test_question_count": len(test_questions),
            "rrf_k_values": ",".join(map(str, RRF_K_VALUES)),
            "top_k_values": ",".join(map(str, TOP_K_VALUES)),
            "selection_metric": "ndcg_at_5,hit_rate_at_5,mrr_at_5,smaller_rrf_k",
        })
        mlflow.log_dict(manifest, "study/manifest.json")
        candidates = []
        for rrf_k in RRF_K_VALUES:
            rankings = _fused_rankings(tune_questions, frozen, rrf_k, dense.chunk_count)
            metrics, rows = _evaluate(tune_questions, rankings)
            with mlflow.start_run(run_name=f"rrf-k-{rrf_k}-tune", nested=True) as child:
                mlflow.log_params({"split": "tune", "method": "rrf", "rrf_k": rrf_k,
                                   "dataset_fingerprint": dataset_fingerprint})
                mlflow.log_metrics(metrics)
                mlflow.log_dict({"rows": rows}, "evaluation/per_case.json")
                candidates.append({"rrf_k": rrf_k, "metrics": metrics, "run_id": child.info.run_id})
        selected = max(candidates, key=lambda item: (
            item["metrics"]["ndcg_at_5"], item["metrics"]["hit_rate_at_5"],
            item["metrics"]["mrr_at_5"], -item["rrf_k"],
        ))
        test_results = {}
        for method in ("bm25", "dense", "rrf"):
            if method == "rrf":
                rankings = _fused_rankings(test_questions, frozen, selected["rrf_k"], dense.chunk_count)
            else:
                rankings = {q["id"]: _document_ranking(frozen[q["id"]][method]) for q in test_questions}
            metrics, rows = _evaluate(test_questions, rankings)
            with mlflow.start_run(run_name=f"{method}-test", nested=True) as child:
                mlflow.log_params({"split": "test", "method": method,
                                   "rrf_k": selected["rrf_k"] if method == "rrf" else "n/a",
                                   "dataset_fingerprint": dataset_fingerprint})
                mlflow.log_metrics(metrics)
                mlflow.log_dict({"rows": rows}, "evaluation/per_case.json")
                test_results[method] = {"metrics": metrics, "run_id": child.info.run_id}
        result = {
            "parent_run_id": parent.info.run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "embedding_model": model,
            "embedding_digest": embedder.model_digest,
            "document_count": len(documents),
            "chunk_count": dense.chunk_count,
            "tune_question_count": len(tune_questions),
            "test_question_count": len(test_questions),
            "selected_rrf_k": selected["rrf_k"],
            "tune_candidates": candidates,
            "test_results": test_results,
        }
        mlflow.log_metrics({f"test_{method}_{name}": value
                            for method, item in test_results.items()
                            for name, value in item["metrics"].items()})
        mlflow.log_dict(result, "comparison/summary.json")
    return result


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv in (["-v"], ["-V"], ["--version"]):
        print(VERSION)
        return 0
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--study-dir", type=Path, default=Path("eval_data/rrf_study"))
    parser.add_argument("--embedding-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--output", type=Path, default=Path("eval_data/results/techqa_rrf_tuning_summary.json"))
    args = parser.parse_args(argv)
    try:
        result = run(args.study_dir, args.embedding_model, args.ollama_url, args.tracking_uri)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        print("help: Verify the prepared study, local Ollama embedding model, and MLflow tracking URI")
        return 1
    print(f"run_id: {result['parent_run_id']}")
    print(f"selected_rrf_k: {result['selected_rrf_k']}")
    print("test[3]{method,hit_rate_at_1,mrr_at_5,ndcg_at_5}:")
    for method, item in result["test_results"].items():
        metrics = item["metrics"]
        print(f"  {method},{metrics['hit_rate_at_1']:.4f},{metrics['mrr_at_5']:.4f},{metrics['ndcg_at_5']:.4f}")
    print(f"summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
