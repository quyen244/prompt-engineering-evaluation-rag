"""Pure-Python deterministic retrieval metrics.

The functions accept ranked IDs and a set of relevant IDs.  They perform no
normalization, fuzzy matching, or model calls: document identity is supplied
by the corpus/retriever contract.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def _top(retrieved: Sequence[str], k: int) -> list[str]:
    _validate_k(k)
    return list(retrieved[:k])


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return float(bool(set(_top(retrieved, k)) & set(relevant)))


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    top = _top(retrieved, k)
    if not top:
        return 0.0
    relevant_ids = set(relevant)
    return sum(doc_id in relevant_ids for doc_id in top) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    return sum(doc_id in relevant_ids for doc_id in _top(retrieved, k)) / len(relevant_ids)


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_ids = set(relevant)
    for rank, doc_id in enumerate(_top(retrieved, k), start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k, with an IDCG capped at available relevance."""
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    hits = [doc_id in relevant_ids for doc_id in _top(retrieved, k)]
    dcg = sum((1.0 / __import__("math").log2(rank + 1)) for rank, hit in enumerate(hits, 1) if hit)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / __import__("math").log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


# Convenient alias for callers that use the conventional spelling.
ndcg = ndcg_at_k
