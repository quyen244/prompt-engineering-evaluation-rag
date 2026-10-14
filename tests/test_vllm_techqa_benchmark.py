"""Offline checks for the frozen TechQA vLLM benchmark inputs and aggregates."""
import unittest

from scripts.vllm_techqa_benchmark import cases, summary


class VllmBenchmarkTests(unittest.TestCase):
    def test_frozen_dense_top_one_matches_all_test_labels(self):
        rows = cases()
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["id"] for row in rows}), 20)
        self.assertTrue(all(row["chunk_id"].startswith(row["doc_id"] + "::chunk-") for row in rows))

    def test_summary_uses_only_successful_requests(self):
        rows = [
            {"error": None, "latency_s": 2.0, "ttft_s": 1.0, "output_tokens": 10},
            {"error": "timeout", "latency_s": 20.0, "ttft_s": None, "output_tokens": 0},
        ]
        details = {"wall_s": 4.0, "gpu_peak_mib": 5000.0,
                   "server_metric_delta": {"prefix_cache_queries_total": 100.0,
                                           "prefix_cache_hits_total": 25.0,
                                           "request_queue_time_seconds_sum": 1.0,
                                           "request_queue_time_seconds_count": 2.0}}
        result = summary(rows, details)
        self.assertEqual(result["success"], 1)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["latency_mean_s"], 2.0)
        self.assertEqual(result["output_tps"], 2.5)
        self.assertEqual(result["prefix_hit_ratio"], 0.25)
        self.assertEqual(result["server_queue_mean_s"], 0.5)


if __name__ == "__main__":
    unittest.main()
