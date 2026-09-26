"""Offline contracts for the isolated TechQA API collection."""
import unittest

from scripts.techqa_index import _load_dataset, _plan, _verify


class TechQAIndexTests(unittest.TestCase):
    def test_dataset_plan_is_complete_and_deterministic(self):
        documents, questions, corpus_hash, _ = _load_dataset()
        collection, points = _plan(documents, corpus_hash, "qwen3-embedding:0.6b", 850, 120)
        self.assertEqual(len(documents), 100)
        self.assertEqual(len(questions), 20)
        self.assertTrue(collection.startswith("techqa_eval_"))
        self.assertEqual({item["payload"]["document_id"] for item in points}, {item["doc_id"] for item in documents})
        self.assertEqual(len({item["id"] for item in points}), len(points))
        self.assertEqual(collection, _plan(documents, corpus_hash, "qwen3-embedding:0.6b", 850, 120)[0])
        _verify(points, points)

    def test_collection_check_rejects_extra_points(self):
        documents, _, corpus_hash, _ = _load_dataset()
        _, points = _plan(documents, corpus_hash, "qwen3-embedding:0.6b", 850, 120)
        with self.assertRaises(ValueError):
            _verify(points, points[:-1])


if __name__ == "__main__":
    unittest.main()
