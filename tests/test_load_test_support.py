import unittest
from types import SimpleNamespace

from scripts.load_test_support import document_fingerprint, query_payload, summarize_stats


class LoadTestSupportTests(unittest.TestCase):
    def test_query_payload_validates_real_documents_and_top_k(self):
        self.assertEqual(
            query_payload("What is Bayes theorem?", ["doc-a"], 5),
            {"question": "What is Bayes theorem?", "documentIds": ["doc-a"], "topK": 5},
        )
        with self.assertRaises(ValueError):
            query_payload("question", [], 5)
        with self.assertRaises(ValueError):
            query_payload("question", ["doc-a"], 51)

    def test_document_fingerprint_is_order_independent(self):
        self.assertEqual(document_fingerprint(["doc-a", "doc-b"]), document_fingerprint(["doc-b", "doc-a"]))

    def test_summary_uses_locust_percentiles_and_actual_request_counts(self):
        stats = SimpleNamespace(
            num_requests=20,
            num_failures=2,
            avg_response_time=1500,
            min_response_time=100,
            max_response_time=3200,
            get_response_time_percentile=lambda percentile: {0.5: 1300, 0.95: 3000, 0.99: 3150}[percentile],
        )
        result = summarize_stats(stats, duration_seconds=10, user_count=4)
        self.assertEqual(result["request_count"], 20)
        self.assertEqual(result["failure_rate"], 0.1)
        self.assertEqual(result["requests_per_second"], 2)
        self.assertEqual(result["response_time_p95_ms"], 3000)


if __name__ == "__main__":
    unittest.main()
