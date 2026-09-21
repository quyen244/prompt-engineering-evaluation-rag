import json
import tempfile
import unittest
from pathlib import Path

from src.telemetry.events import DocumentExperimentEvent, RequestEvent
from src.telemetry.experiment_history import ExperimentHistoryWriter
from src.telemetry.metrics import PROMETHEUS_LABELS
from src.telemetry.redaction import dumps_event, redact_event


class RedactionTests(unittest.TestCase):
    def test_question_answer_and_secret_never_survive(self):
        raw_question = "What is the confidential customer address?"
        raw_answer = "The answer contains private source material."
        encoded = dumps_event({
            "question": raw_question,
            "answer": raw_answer,
            "authorization": "Bearer top-secret",
            "request_id": "req-1",
        })
        self.assertNotIn(raw_question, encoded)
        self.assertNotIn(raw_answer, encoded)
        self.assertNotIn("top-secret", encoded)
        self.assertIn("question_sha256", encoded)
        self.assertIn("question_length", encoded)

    def test_nested_values_are_redacted(self):
        value = redact_event({"metadata": {"api_key": "secret", "query": "private"}})
        self.assertEqual(value["metadata"]["api_key"], "[REDACTED]")
        self.assertNotIn("private", json.dumps(value))


class SchemaTests(unittest.TestCase):
    def test_request_contract_has_traceability_fields_without_content(self):
        event = RequestEvent(
            retrieval_strategy="bm25-plus-reranker", prompt_version="prompt-1",
            model_version="model-1", dataset_split="test", latency_ms=12.0,
            prompt_tokens=10, completion_tokens=3, cost_usd=0.01,
        ).to_dict()
        for name in ("request_id", "run_id", "retrieval_strategy", "prompt_version", "model_version", "dataset_split", "latency_ms", "prompt_tokens", "cost_usd", "outcome"):
            self.assertIn(name, event)
        self.assertNotIn("question", event)
        self.assertEqual(event["retrieval_strategy"], "bm25_reranker")

    def test_document_history_is_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            writer = ExperimentHistoryWriter(path)
            event = DocumentExperimentEvent(
                document_id="opaque-doc-1", operation="evaluate", retrieval_strategy="dense",
                prompt_version="prompt-1", model_version="model-1", dataset_split="validation",
            )
            self.assertTrue(writer.append(event))
            self.assertTrue(writer.append(event))
            self.assertEqual(len(path.read_text().splitlines()), 2)


class CardinalityTests(unittest.TestCase):
    def test_metric_label_vocabulary_is_bounded(self):
        forbidden = {"request_id", "run_id", "trace_id", "user_id", "question", "answer", "document_id"}
        labels = set(PROMETHEUS_LABELS)
        self.assertTrue(labels.isdisjoint(forbidden))
        self.assertEqual(PROMETHEUS_LABELS["retrieval_strategy"], {"bm25_only", "bm25_reranker", "dense", "hybrid"})


if __name__ == "__main__":
    unittest.main()
