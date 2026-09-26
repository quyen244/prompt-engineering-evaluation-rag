"""Small offline checks for TechQA data integrity and retrieval scoring."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from scripts.techqa_retrieval import build_retriever, evaluate_retriever, load_dataset, run_ablation


class _FakeMLflow:
    def __init__(self) -> None:
        self.logged_artifacts: list[str] = []
        self.logged_metrics: list[dict[str, float]] = []
        self.logged_params: list[dict[str, object]] = []
        self.run_count = 0

    def set_tracking_uri(self, value: str) -> None:
        self.tracking_uri = value

    def set_experiment(self, value: str) -> None:
        self.experiment = value

    def start_run(self, run_name: str | None = None, run_id: str | None = None):
        if run_id is None:
            self.run_count += 1
            run_id = f"run-{self.run_count}"
        class _Run:
            info = SimpleNamespace(run_id=run_id)
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
        return _Run()

    def set_tags(self, values: dict[str, str]) -> None:
        self.tags = values

    def log_params(self, values: dict[str, object]) -> None:
        self.logged_params.append(values)

    def log_metrics(self, values: dict[str, float]) -> None:
        self.logged_metrics.append(values)

    def log_dict(self, value: dict[str, object], artifact_file: str) -> None:
        self.logged_artifacts.append(artifact_file)


class _KeywordEmbedder:
    model_digest = "offline-test"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if "lateanswer" in text else [1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


class TechQARetrievalTest(TestCase):
    def test_checked_in_dataset_has_all_gold_documents(self) -> None:
        documents, questions, fingerprint = load_dataset()
        self.assertEqual(len(documents), 100)
        self.assertEqual(len(questions), 20)
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(len({document["doc_id"] for document in documents}), 100)
        self.assertTrue({question["gold_doc_id"] for question in questions} <=
                        {document["doc_id"] for document in documents})

    def test_bm25_retrieval_and_metrics_use_gold_id(self) -> None:
        docs = [{"doc_id": "a", "title": "orbit engine", "text": "orbit engine pressure"},
                {"doc_id": "b", "title": "garden", "text": "tomato flowers"}]
        questions = [{"id": "q1", "question": "orbit engine pressure", "ground_truth": "fixed",
                      "gold_doc_id": "a"}]
        retriever = build_retriever(docs, method="bm25")
        self.assertEqual(retriever.retrieve("orbit engine pressure", 1)[0]["doc_id"], "a")
        metrics, rows = evaluate_retriever(retriever, questions, 1)
        self.assertEqual(metrics["hit_rate_at_1"], 1.0)
        self.assertEqual(metrics["mrr_at_1"], 1.0)
        self.assertEqual(rows[0]["retrieved_doc_ids"], ["a"])

    def test_late_evidence_is_retrieved_from_long_document(self) -> None:
        docs = [{"doc_id": "a", "title": "Long note", "text": "filler " * 500 + " lateanswer"},
                {"doc_id": "b", "title": "Other note", "text": "ordinary material"}]
        retriever = build_retriever(docs, method="dense", embedder=_KeywordEmbedder())
        result = retriever.retrieve("lateanswer", 1)[0]
        self.assertEqual(result["doc_id"], "a")
        self.assertIn("lateanswer", result["context_text"])
        self.assertIn("lateanswer", result["text"])
        self.assertGreater(retriever.chunk_count, len(docs))

    def test_missing_gold_fails_before_retrieval(self) -> None:
        with TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            questions = Path(directory) / "questions.json"
            corpus.write_text(json.dumps({"doc_id": "a", "title": "A", "text": "text"}) + "\n", encoding="utf-8")
            questions.write_text(json.dumps([{"id": "q", "question": "what", "ground_truth": "answer",
                                              "gold_doc_id": "missing"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gold_doc_id absent"):
                load_dataset(corpus, questions)

    def test_ablation_tracks_cases_and_writes_champion(self) -> None:
        docs = [{"doc_id": "a", "title": "orbit engine", "text": "orbit engine pressure"},
                {"doc_id": "b", "title": "garden", "text": "tomato flowers"}]
        questions = [{"id": "q1", "question": "orbit engine pressure", "ground_truth": "fixed",
                      "gold_doc_id": "a"}]
        tracking = _FakeMLflow()
        with TemporaryDirectory() as directory:
            champion_path = Path(directory) / "champion.json"
            runs = run_ablation(docs, questions, "dataset-digest", methods=["bm25"],
                                top_ks=[1, 2], mlflow_module=tracking, champion_path=champion_path)
            champion = json.loads(champion_path.read_text(encoding="utf-8"))
        self.assertEqual(len(runs), 2)
        self.assertEqual(champion["top_k"], 1)
        self.assertEqual(champion["retrieval_run_id"], "run-1")
        self.assertEqual(champion["dataset_fingerprint"], "dataset-digest")
        self.assertEqual(len(tracking.logged_metrics), 2)
        self.assertEqual(tracking.logged_artifacts.count("evaluation/per_case.json"), 2)
        self.assertIn("comparison/champion.json", tracking.logged_artifacts)
