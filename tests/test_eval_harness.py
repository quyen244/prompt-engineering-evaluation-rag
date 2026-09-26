import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.eval.dataset import load_cases
from src.eval.metrics.retrieval import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.eval.runner import EvalConfig, paired_compare, run_eval
from src.eval.schema import EvalCase


class FakeRetriever:
    def invoke(self, question):
        return [SimpleNamespace(page_content="context", metadata={"document_id": "not-relevant"})]


def cases() -> list[EvalCase]:
    return [EvalCase(
        case_id=f"case-{index}", question=f"Question {index}?",
        expected_answer="Reference answer", relevant_document_ids=(f"gold-{index}",),
        source="test_fixture",
    ) for index in range(20)]


class EvaluationHarnessTests(unittest.TestCase):
    def test_cases_load_from_an_explicit_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(case.to_dict()) for case in cases()), encoding="utf-8")
            loaded = load_cases(path)
            self.assertEqual(len(loaded), 20)
            self.assertEqual([case.case_id for case in loaded], [case.case_id for case in cases()])

    def test_missing_source_fails_without_fallback_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "evaluation data is missing"):
                load_cases(Path(directory) / "missing.jsonl")

    def test_retrieval_metrics(self):
        ranked = ["wrong", "gold", "other"]
        relevant = {"gold"}
        self.assertEqual(hit_rate_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(mrr_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, 2), 1 / 1.584962500721156)

    def test_run_writes_per_query_and_append_only_history(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            output = Path(directory) / "results"
            first = run_eval(EvalConfig(name="a"), cases(), FakeRetriever(), output_dir=output, history_path=history)
            size = history.stat().st_size
            second = run_eval(EvalConfig(name="b"), cases(), FakeRetriever(), output_dir=output, history_path=history)
            self.assertGreater(history.stat().st_size, size)
            self.assertEqual(len(first.query_results), 20)
            self.assertEqual(len(list(output.glob("*.jsonl"))), 2)
            self.assertEqual(paired_compare(first, second)["case_count"], 20)


if __name__ == "__main__":
    unittest.main()
