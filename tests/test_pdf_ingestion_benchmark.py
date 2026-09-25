from __future__ import annotations

from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark_pdf_ingestion import run_benchmark


class FakeMlflow:
    def __init__(self):
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.tags = []
        self.runs = 0

    def set_tracking_uri(self, uri):
        self.tracking_uri = uri

    def set_experiment(self, name):
        self.experiment = name

    @contextmanager
    def start_run(self, run_name, nested=False):
        self.runs += 1
        yield SimpleNamespace(info=SimpleNamespace(run_id=f"run-{self.runs}"))

    def set_tags(self, tags):
        self.tags.append(tags)

    def log_params(self, params):
        self.params.append(params)

    def log_metrics(self, metrics):
        self.metrics.append(metrics)

    def log_dict(self, value, path):
        self.artifacts.append((value, path))


class PdfIngestionBenchmarkTests(unittest.TestCase):
    def test_logs_paired_real_run_measurements_and_fingerprint(self):
        mlflow = FakeMlflow()
        calls = []

        def sync(filename, content):
            calls.append("sync")
            return {"end_to_end_seconds": 10.0, "result": {"documentId": "doc-a", "chunkCount": 8}}

        def async_(filename, content, poll_seconds, timeout_seconds):
            calls.append("celery")
            return {
                "enqueue_seconds": 0.2,
                "end_to_end_seconds": 12.0,
                "worker_processing_seconds": 11.5,
                "result": {"documentId": "doc-a", "chunkCount": 8},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf = Path(temp_dir) / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.7 real test bytes")
            result = run_benchmark(
                pdf_path=pdf, trials=2, experiment_name="test", tracking_uri="sqlite:///test.db",
                sync_ingest=sync, async_ingest=async_, mlflow_module=mlflow,
            )

        self.assertEqual(calls, ["sync", "celery", "celery", "sync"])
        self.assertEqual(result["trial_count"], 2)
        self.assertAlmostEqual(result["mean_end_to_end_delta_seconds"], 2.0)
        self.assertAlmostEqual(result["mean_end_to_end_delta_percent"], 20.0)
        self.assertEqual(len(result["pdf_sha256"]), 64)
        self.assertTrue(any(path == "comparison/summary.json" for _, path in mlflow.artifacts))
        self.assertEqual(mlflow.tracking_uri, "sqlite:///test.db")

    def test_rejects_non_pdf_without_logging_a_result(self):
        mlflow = FakeMlflow()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "not-a-pdf.pdf"
            path.write_bytes(b"not pdf")
            with self.assertRaisesRegex(ValueError, "PDF header"):
                run_benchmark(
                    pdf_path=path, trials=1, experiment_name="test", tracking_uri="sqlite:///test.db",
                    sync_ingest=lambda *_: {}, async_ingest=lambda *_: {}, mlflow_module=mlflow,
                )
        self.assertEqual(mlflow.runs, 0)


if __name__ == "__main__":
    unittest.main()
