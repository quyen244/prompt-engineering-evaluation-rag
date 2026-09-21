import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.eval.dataset import load_cases, load_manifest, validate_project_cases, validate_split_disjointness
from src.eval.metrics.retrieval import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.eval.runner import EvalConfig, mode_cases, paired_compare, run_eval


ROOT = Path(__file__).resolve().parents[1]


class FakeRetriever:
    def invoke(self, question):
        return [SimpleNamespace(page_content="context", metadata={"document_id": "not-relevant"})]


class EvaluationHarnessTests(unittest.TestCase):
    def test_manifest_counts_and_disjoint_splits(self):
        manifest = load_manifest(ROOT / "eval_data/manifests/evaluation_manifest.json")
        validate_split_disjointness(manifest)
        self.assertEqual(manifest["project_source"]["count"], 20)
        self.assertEqual(manifest["huggingface"]["splits"]["improvement"]["count"], 30)
        self.assertEqual(manifest["huggingface"]["splits"]["validation"]["count"], 30)

    def test_project_cases_are_real_and_source_spans_match(self):
        cases = load_cases(ROOT / "eval_data/project_source_cases.jsonl")
        self.assertEqual(len(cases), 20)
        self.assertTrue(validate_project_cases(cases, ROOT).valid)
        self.assertTrue(all(case.source == "project" for case in cases))

    def test_retrieval_metrics(self):
        ranked = ["wrong", "gold", "other"]
        relevant = {"gold"}
        self.assertEqual(hit_rate_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(ranked, relevant, 2), 1.0)
        self.assertEqual(mrr_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, 2), 1 / 1.584962500721156)

    def test_run_writes_per_query_and_append_only_history(self):
        cases = mode_cases("smoke", root=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            output = Path(directory) / "results"
            first = run_eval(EvalConfig(name="a"), cases, FakeRetriever(), output_dir=output, history_path=history)
            size = history.stat().st_size
            second = run_eval(EvalConfig(name="b"), cases, FakeRetriever(), output_dir=output, history_path=history)
            self.assertGreater(history.stat().st_size, size)
            self.assertEqual(len(first.query_results), 20)
            self.assertEqual(len(list(output.glob("*.jsonl"))), 2)
            self.assertEqual(paired_compare(first, second)["case_count"], 20)


if __name__ == "__main__":
    unittest.main()
