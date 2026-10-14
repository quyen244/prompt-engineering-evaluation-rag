"""Offline contracts for the frozen-rank RRF comparison."""
from __future__ import annotations

import unittest

from scripts.run_techqa_rrf_tuning import _document_ranking, _evaluate, _fused_rankings
from src.rag.models import RetrievedSource


def source(doc_id: str, chunk_id: str, rank: int, method: str) -> RetrievedSource:
    return RetrievedSource(chunk_id, doc_id, float(10 - rank),
                           {"document_id": doc_id}, method)


class RRFTuningTests(unittest.TestCase):
    def test_fusion_reranks_frozen_passages_and_deduplicates_documents(self) -> None:
        question = {"id": "q1", "question": "sample", "gold_doc_id": "gold"}
        frozen = {"q1": {
            "bm25": [source("other", "other-1", 1, "bm25"),
                     source("gold", "gold-1", 2, "bm25"),
                     source("gold", "gold-2", 3, "bm25")],
            "dense": [source("gold", "gold-1", 1, "dense"),
                      source("other", "other-1", 2, "dense"),
                      source("gold", "gold-2", 3, "dense")],
        }}
        ranked = _fused_rankings([question], frozen, rrf_k=1, chunk_count=3)
        self.assertEqual([row["doc_id"] for row in ranked["q1"]], ["gold", "other"])
        metrics, rows = _evaluate([question], ranked)
        self.assertEqual(metrics["hit_rate_at_1"], 1.0)
        self.assertEqual(metrics["ndcg_at_5"], 1.0)
        self.assertEqual(rows[0]["retrieved_chunk_ids_at_5"], ["gold-1", "other-1"])

    def test_document_ranking_keeps_first_passage_for_each_document(self) -> None:
        ranked = _document_ranking([
            source("a", "a-1", 1, "dense"),
            source("a", "a-2", 2, "dense"),
            source("b", "b-1", 3, "dense"),
        ])
        self.assertEqual([(row["doc_id"], row["chunk_id"]) for row in ranked],
                         [("a", "a-1"), ("b", "b-1")])


if __name__ == "__main__":
    unittest.main()
