import pytest

from src.rag.lineage import (
    BudgetMetadata,
    DatasetIdentity,
    ModelIdentity,
    PromptVariableError,
    PromptVersion,
    PromotionEvidence,
    PromotionRefused,
    RetrievalIdentity,
    build_manifest,
    digest,
    require_promotion,
)


def _manifest(prompt_text="Answer {{question}} using {{context}}"):
    return build_manifest(
        retrieval=RetrievalIdentity("hybrid", 5, bm25_digest="bm25", hybrid_digest="rrf"),
        prompt=PromptVersion("rag-answer", 1, prompt_text),
        model=ModelIdentity("openrouter", "model-a", "2025-01"),
        dataset=DatasetIdentity("heldout", digest(["case-a", "case-b"]), "validation", 2),
        budget=BudgetMetadata(20, cache_enabled=True, cache_namespace="test"),
        git_commit="abc123",
    )


def test_manifest_identity_changes_with_prompt_and_is_content_addressed():
    first = _manifest()
    second = _manifest("Answer carefully: {{question}} using {{context}}")
    assert first.identity == first.manifest_digest
    assert first.identity == _manifest().identity
    assert first.identity != second.identity
    with pytest.raises(TypeError):
        first.retrieval.strategy = "bm25"


def test_prompt_variables_are_validated_before_invocation():
    prompt = PromptVersion("qa", 1, "Q={{ question }} C={{context}}")
    assert prompt.render(question="why", context="evidence") == "Q=why C=evidence"
    with pytest.raises(PromptVariableError, match="context"):
        prompt.render(question="why")
    with pytest.raises(PromptVariableError):
        PromptVersion("qa", 1, "Q={{question}}", variables=frozenset({"question", "context"}))


def test_promotion_refuses_incomplete_heldout_evidence():
    evidence = PromotionEvidence(
        baseline_score=0.5,
        candidate_score=0.8,
        heldout_case_count=0,
        heldout_evaluated=False,
        retrieval_frozen=True,
        candidates_recorded=True,
        real_data=True,
        dataset_digest="real-digest",
        candidate_run_id="run-1",
    )
    with pytest.raises(PromotionRefused, match="heldout_evaluation"):
        require_promotion(evidence)


def test_promotion_does_not_claim_improvement_without_real_data():
    evidence = PromotionEvidence(
        baseline_score=0.5,
        candidate_score=0.8,
        heldout_case_count=2,
        heldout_evaluated=True,
        retrieval_frozen=True,
        candidates_recorded=True,
        real_data=False,
        dataset_digest="",
        candidate_run_id="run-1",
    )
    with pytest.raises(PromotionRefused, match="real_dataset"):
        require_promotion(evidence)
