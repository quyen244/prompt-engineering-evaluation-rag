"""Bounded vocabulary shared by all telemetry surfaces.

Values in this module are intentionally finite: they are safe for metric labels and
make schema drift visible during review.
"""

RETRIEVAL_STRATEGIES = frozenset({"bm25_only", "bm25_reranker", "dense", "hybrid"})
RETRIEVAL_STAGES = frozenset({
    "upload", "parse", "chunk", "embed", "index", "retrieve", "rerank",
    "generate", "query", "ingest",
})
OUTCOMES = frozenset({"success", "no_answer", "error", "timeout", "cancelled"})
DATASET_SPLITS = frozenset({"train", "validation", "test", "production", "unknown"})
EVENT_TYPES = frozenset({"request", "retrieval", "generation", "evaluation", "error", "document_experiment"})
QUALITY_METRICS = frozenset({
    "faithfulness", "correctness", "answer_relevancy", "hit_rate", "mrr",
    "precision", "recall", "ndcg",
})
GENERATION_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_call", "unknown"})
ERROR_TYPES = frozenset({"timeout", "upstream", "validation", "retrieval", "generation", "unknown"})

# Bound event payloads before they reach a durable sink.  These are deliberately
# small enough to make an accidental high-cardinality diagnostic payload visible.
MAX_EVENT_STRING_LENGTH = 128
MAX_EVENT_COLLECTION_ITEMS = 32
MAX_EVENT_NESTING = 4

# These are the only values allowed in Prometheus labels. IDs belong on spans/events.
PROMETHEUS_LABELS = {
    "stage": RETRIEVAL_STAGES,
    "retrieval_strategy": RETRIEVAL_STRATEGIES,
    "outcome": OUTCOMES,
    "token_type": frozenset({"prompt", "completion"}),
    "error_type": ERROR_TYPES,
    "quality_metric": QUALITY_METRICS,
    "dataset_split": DATASET_SPLITS,
}

SENSITIVE_KEY_PARTS = (
    "api_key", "apikey", "authorization", "password", "passwd", "secret",
    "access_token", "refresh_token", "api_token", "auth_token", "bearer_token", "private_key", "credential", "cookie", "set-cookie",
    "question", "answer", "prompt", "completion",
)
