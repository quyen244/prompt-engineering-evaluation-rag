import json
import tempfile
import unittest
from pathlib import Path

from src.telemetry.events import (
    DocumentExperimentEvent, ErrorEvent, EvaluationEvent, GenerationEvent,
    RequestEvent, RetrievalEvent,
)
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

    def test_unknown_text_and_collections_are_bounded(self):
        value = redact_event({"diagnostic": "customer@example.test", "items": list(range(40))})
        self.assertNotIn("customer@example.test", json.dumps(value))
        self.assertEqual(value["diagnostic"]["length"], len("customer@example.test"))
        self.assertTrue(value["items"]["truncated"])
        self.assertEqual(len(value["items"]["items"]), 32)


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

    def test_stage_events_preserve_lineage_without_raw_content(self):
        common = {"retrieval_strategy": "hybrid", "prompt_version": "p-1", "model_version": "m-1", "dataset_split": "test"}
        events = [
            RetrievalEvent(**common, retrieved_document_count=3, top_score=0.8, context_tokens=100),
            GenerationEvent(**common, prompt_tokens=100, completion_tokens=20, finish_reason="stop"),
            EvaluationEvent(**common, quality_metric="faithfulness", quality_score=0.9, quality_threshold=0.8),
            ErrorEvent(**common, stage="generate", error_type="upstream", error_code="UPSTREAM_503", retryable=True),
        ]
        self.assertEqual([event.to_dict()["event_type"] for event in events], ["retrieval", "generation", "evaluation", "error"])
        for event in events:
            payload = event.to_dict()
            self.assertEqual(payload["prompt_version"], "p-1")
            self.assertEqual(payload["model_version"], "m-1")
            self.assertNotIn("question", payload)

    def test_evaluation_requires_real_bounded_values_when_supplied(self):
        with self.assertRaises(ValueError):
            EvaluationEvent(
                retrieval_strategy="hybrid", prompt_version="p-1", model_version="m-1",
                dataset_split="test", quality_metric="faithfulness", quality_score=1.1,
            )


class CardinalityTests(unittest.TestCase):
    def test_metric_label_vocabulary_is_bounded(self):
        forbidden = {"request_id", "run_id", "trace_id", "user_id", "question", "answer", "document_id"}
        labels = set(PROMETHEUS_LABELS)
        self.assertTrue(labels.isdisjoint(forbidden))
        self.assertEqual(PROMETHEUS_LABELS["retrieval_strategy"], {"bm25_only", "bm25_reranker", "dense", "hybrid"})


if __name__ == "__main__":
    unittest.main()
