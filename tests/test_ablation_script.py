import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_retrieval_ablation import build_configs, run_ablation
from scripts.project_embedding_factory import DenseProjectRetriever
from src.eval.runner import EvalConfig
from src.eval.schema import EvalCase


ROOT = Path(__file__).resolve().parents[1]


class FakeMlflow:
    def __init__(self):
        self.runs = []
        self.current = None

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, run_name=None, run_id=None):
        outer = self

        class Run:
            def __enter__(self):
                self.previous = outer.current
                self.record = next((item for item in outer.runs if item["id"] == run_id), None)
                if self.record is None:
                    self.record = {"id": run_id or f"run-{len(outer.runs) + 1}", "artifacts": {}}
                    outer.runs.append(self.record)
                outer.current = self.record
                return SimpleNamespace(info=SimpleNamespace(run_id=self.record["id"]))

            def __exit__(self, *args):
                outer.current = self.previous

        return Run()

    def set_tags(self, tags):
        self.current["tags"] = tags

    def log_params(self, params):
        self.current["params"] = params

    def log_metrics(self, metrics):
        self.current["metrics"] = metrics

    def log_dict(self, value, artifact_file):
        self.current["artifacts"][artifact_file] = value


class FakeRetriever:
    def invoke(self, question):
        return [SimpleNamespace(page_content="result", metadata={"document_id": "not-relevant"})]


class AblationScriptTests(unittest.TestCase):
    def test_builds_cartesian_embedding_and_top_k_matrix(self):
        configs = build_configs(
            [5, 10, 20], ["huggingface:intfloat/multilingual-e5-base", "ollama:qwen3-embedding:0.6b"],
            "dense",
        )
        self.assertEqual(len(configs), 6)
        self.assertEqual({config.top_k for config in configs}, {5, 10, 20})
        self.assertEqual({config.embedding for config in configs}, {
            "huggingface:intfloat/multilingual-e5-base", "ollama:qwen3-embedding:0.6b",
        })

    def test_dense_retriever_deduplicates_chunks_to_stable_source_ids(self):
        import numpy as np

        class Embedder:
            def encode(self, _texts):
                return [[1.0, 0.0]]

        retriever = DenseProjectRetriever(
            EvalConfig(name="dense", top_k=2, embedding="ollama:test"),
            ["project:a.md", "project:a.md", "project:b.md"],
            ["a first chunk", "a better chunk", "b chunk"],
            np.asarray([[0.8, 0.2], [1.0, 0.0], [0.9, 0.1]], dtype=np.float32),
            Embedder(),
        )
        documents = retriever.invoke("question")
        self.assertEqual([document.metadata["document_id"] for document in documents], ["project:a.md", "project:b.md"])
        self.assertEqual(documents[0].page_content, "a better chunk")

    def test_paired_runs_are_logged_to_injected_mlflow(self):
        cases = [EvalCase(
            case_id=f"case-{index}", question=f"Question {index}?",
            expected_answer="Reference answer", relevant_document_ids=(f"gold-{index}",),
            source="test_fixture",
        ) for index in range(20)]
        mlflow = FakeMlflow()
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.run_retrieval_ablation.load_manifest",
            return_value={"schema_version": 1, "fixture": "unit-test"},
        ):
            study_id, reports, comparison = run_ablation(
                cases=cases,
                configs=[EvalConfig(name="top-3", top_k=3), EvalConfig(name="top-5", top_k=5)],
                retriever_factory=lambda config, options: FakeRetriever(),
                mlflow_module=mlflow,
                mode="smoke",
                experiment_name="test-ablation",
                tracking_uri="sqlite:///test.db",
                output_dir=Path(directory) / "results",
                history_path=Path(directory) / "history.jsonl",
                manifest_path=Path(directory) / "unused-manifest.json",
            )

        self.assertEqual(len(reports), 2)
        self.assertEqual(comparison["case_count"], 20)
        self.assertTrue(study_id)
        self.assertEqual(len(mlflow.runs), 2)
        self.assertIn("evaluation/per_case.json", mlflow.runs[0]["artifacts"])
        self.assertIn("comparison/paired_deltas.json", mlflow.runs[0]["artifacts"])
        self.assertEqual(mlflow.runs[0]["metrics"]["hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
