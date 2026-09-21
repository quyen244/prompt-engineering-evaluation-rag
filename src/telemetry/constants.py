"""Bounded vocabulary shared by all telemetry surfaces.

Values in this module are intentionally finite: they are safe for metric labels and
make schema drift visible during review.
"""

RETRIEVAL_STRATEGIES = frozenset({"bm25_only", "bm25_reranker", "dense", "hybrid"})
RETRIEVAL_STAGES = frozenset({"retrieve", "rerank", "generate", "query", "ingest"})
OUTCOMES = frozenset({"success", "no_answer", "error", "timeout", "cancelled"})
DATASET_SPLITS = frozenset({"train", "validation", "test", "production", "unknown"})
EVENT_TYPES = frozenset({"request", "document_experiment"})

# These are the only values allowed in Prometheus labels. IDs belong on spans/events.
PROMETHEUS_LABELS = {
    "stage": RETRIEVAL_STAGES,
    "retrieval_strategy": RETRIEVAL_STRATEGIES,
    "outcome": OUTCOMES,
    "token_type": frozenset({"prompt", "completion"}),
    "error_type": frozenset({"timeout", "upstream", "validation", "retrieval", "generation", "unknown"}),
}

SENSITIVE_KEY_PARTS = (
    "api_key", "apikey", "authorization", "password", "passwd", "secret",
    "access_token", "refresh_token", "api_token", "auth_token", "bearer_token", "private_key", "credential", "cookie", "set-cookie",
    "question", "answer", "prompt", "completion",
)
