from src.rag.lineage import BudgetMetadata, PromptVersion
from src.rag.optimization.gepa_adapter import (
    GepaLangChainAdapter,
    GepaSetupError,
    freeze_retrieval,
)


class FakeRunnable:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"answer": f"{payload['prompt']} -> answer"}


def _cases():
    return [
        {"case_id": "a", "inputs": {"question": "What is A?"}},
        {"case_id": "b", "inputs": {"question": "What is B?"}},
    ]


def test_frozen_retrieval_is_materialized_once_and_never_called_by_adapter():
    retrieve_calls = []

    def retrieve(question):
        retrieve_calls.append(question)
        return {"context": f"evidence for {question}", "document_ids": [question[-2:]]}

    snapshots = freeze_retrieval(_cases(), retrieve)
    assert len(retrieve_calls) == 2
    runnable = FakeRunnable()
    adapter = GepaLangChainAdapter(
        runnable, snapshots,
        budget=BudgetMetadata(10, cache_enabled=True, cache_namespace="test"),
    )
    prompt = PromptVersion("qa", 1, "Answer {{question}} from {{context}}")
    first = adapter.predict(prompt, "What is A?", candidate_id="baseline")
    second = adapter.predict(prompt, "What is A?", candidate_id="baseline")
    assert first == second
    assert len(retrieve_calls) == 2
    assert len(runnable.calls) == 1
    assert adapter.cache_stats == {"hits": 1, "misses": 1}
    assert adapter.frozen_retrieval_contract.snapshot_count == 2


def test_selection_records_observed_scores_and_frozen_snapshot_identity():
    snapshots = freeze_retrieval(_cases(), lambda question: f"context:{question}")
    adapter = GepaLangChainAdapter(
        FakeRunnable(), snapshots,
        budget=BudgetMetadata(20),
    )
    baseline = PromptVersion("qa", 1, "baseline {{question}} {{context}}")
    candidate = PromptVersion("qa", 2, "better {{question}} {{context}}")

    def scorer(output, _case):
        return 1.0 if output.startswith("better") else 0.25

    baseline_result = adapter.validate_heldout(
        baseline, _cases(), scorer, candidate_id="baseline", dataset_digest="heldout-v1"
    )
    candidate_result = adapter.validate_heldout(
        candidate, _cases(), scorer, candidate_id="candidate-2", dataset_digest="heldout-v1"
    )
    selection = adapter.record_selection(
        baseline_result, candidate_result, selection_run_id="run-1"
    )
    assert selection.selected_candidate_id == "candidate-2"
    assert selection.selected_score > selection.baseline_score
    assert selection.selection_digest
    assert selection.candidate_count == 2


def test_heldout_validation_rejects_empty_or_non_numeric_evidence():
    snapshots = freeze_retrieval(_cases(), lambda question: "context")
    adapter = GepaLangChainAdapter(FakeRunnable(), snapshots, budget=BudgetMetadata(3))
    prompt = PromptVersion("qa", 1, "{{question}} {{context}}")
    try:
        adapter.validate_heldout(prompt, [], lambda _output, _case: 1.0,
                                 candidate_id="candidate", dataset_digest="digest")
    except GepaSetupError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("empty held-out data must be rejected")
