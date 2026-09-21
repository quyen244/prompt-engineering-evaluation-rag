"""GEPA adapter for a LangChain Runnable with frozen retrieval evidence.

GEPA is allowed to change the prompt only.  Retrieval results are materialized
before optimization and are never obtained from a retriever while candidates
are being scored.  This prevents prompt candidates from being compared on
different contexts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.rag.lineage import (
    BudgetMetadata,
    CaseArtifactLink,
    PromptVersion,
    digest,
    child_run_name,
)


class GepaSetupError(RuntimeError):
    """An optional optimization dependency or runnable contract is unavailable."""


@dataclass(frozen=True)
class RetrievalSnapshot:
    case_id: str
    question: str
    context: str
    document_ids: tuple[str, ...] = ()
    retrieval_digest: str = ""

    def __post_init__(self) -> None:
        if not self.case_id or not self.question:
            raise ValueError("retrieval snapshots require case_id and question")
        if not self.retrieval_digest:
            object.__setattr__(self, "retrieval_digest", digest({
                "case_id": self.case_id,
                "question": self.question,
                "context": self.context,
                "document_ids": self.document_ids,
            }))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "context": self.context,
            "document_ids": list(self.document_ids),
            "retrieval_digest": self.retrieval_digest,
        }


@dataclass(frozen=True)
class HeldoutValidation:
    candidate_id: str
    dataset_digest: str
    scores: tuple[float, ...]
    case_count: int
    evaluated: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.dataset_digest or self.case_count < 1:
            raise GepaSetupError("held-out validation requires a candidate, dataset digest, and cases")
        if not self.evaluated or len(self.scores) != self.case_count:
            raise GepaSetupError("held-out validation must contain one real score per case")
        if any(isinstance(score, bool) or not isinstance(score, (int, float)) for score in self.scores):
            raise GepaSetupError("held-out validation scores must be numeric")

    @property
    def mean_score(self) -> float:
        if not self.scores:
            raise GepaSetupError("held-out validation has no scores")
        return sum(self.scores) / len(self.scores)


@dataclass(frozen=True)
class GepaSelectionRecord:
    """Auditable record of which immutable candidate won a GEPA comparison."""

    selected_candidate_id: str
    selected_prompt_digest: str
    baseline_candidate_id: str
    baseline_score: float
    selected_score: float
    dataset_digest: str
    selection_split: str
    candidate_count: int
    retrieval_snapshot_digest: str
    selection_run_id: str
    selection_digest: str = ""

    def __post_init__(self) -> None:
        if not self.selected_candidate_id or not self.selected_prompt_digest:
            raise GepaSetupError("selection must identify the selected prompt candidate")
        if not self.baseline_candidate_id or not self.dataset_digest or not self.selection_run_id:
            raise GepaSetupError("selection must identify baseline, dataset, and run")
        if self.candidate_count < 1 or not self.selection_split or not self.retrieval_snapshot_digest:
            raise GepaSetupError("selection must include split, candidate count, and frozen retrieval identity")
        for score in (self.baseline_score, self.selected_score):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise GepaSetupError("selection scores must be numeric observations")
        payload = self.to_dict(include_digest=False)
        expected = digest(payload)
        if self.selection_digest and self.selection_digest != expected:
            raise GepaSetupError("selection digest does not match its immutable contents")
        object.__setattr__(self, "selection_digest", expected)

    def to_dict(self, include_digest: bool = True) -> dict[str, Any]:
        data = {
            "selected_candidate_id": self.selected_candidate_id,
            "selected_prompt_digest": self.selected_prompt_digest,
            "baseline_candidate_id": self.baseline_candidate_id,
            "baseline_score": self.baseline_score,
            "selected_score": self.selected_score,
            "dataset_digest": self.dataset_digest,
            "selection_split": self.selection_split,
            "candidate_count": self.candidate_count,
            "retrieval_snapshot_digest": self.retrieval_snapshot_digest,
            "selection_run_id": self.selection_run_id,
        }
        if include_digest:
            data["selection_digest"] = self.selection_digest
        return data


@dataclass(frozen=True)
class FrozenRetrievalContract:
    """Reproducibility identity for materialized retrieval evidence."""

    snapshot_count: int
    snapshot_digest: str
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.snapshot_count < 1 or self.snapshot_count != len(self.case_ids) or not self.snapshot_digest:
            raise GepaSetupError("frozen retrieval contract is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_count": self.snapshot_count, "snapshot_digest": self.snapshot_digest,
                "case_ids": list(self.case_ids)}


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    prompt_digest: str
    case_id: str
    split: str
    output: str
    result_digest: str
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prompt_digest": self.prompt_digest,
            "case_id": self.case_id,
            "split": self.split,
            "output": self.output,
            "result_digest": self.result_digest,
            "cache_hit": self.cache_hit,
        }


@dataclass
class OptimizationLedger:
    """Append-only candidate/result ledger retained for artifact logging."""

    records: list[CandidateResult] = field(default_factory=list)

    def append(self, result: CandidateResult) -> None:
        self.records.append(result)

    def links(self, run_id: str) -> tuple[CaseArtifactLink, ...]:
        return tuple(CaseArtifactLink(
            case_id=result.case_id,
            split=result.split,
            run_id=run_id,
            artifact_path=f"candidates/{result.candidate_id}/{result.split}/{result.case_id}.json",
            result_digest=result.result_digest,
        ) for result in self.records)

    def jsonl(self) -> str:
        return "".join(json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for record in self.records)


@dataclass
class _Cache:
    values: dict[str, str] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0


class GepaLangChainAdapter:
    """Adapt a LangChain-style ``Runnable.invoke`` to GEPA evaluation.

    ``retrieval_snapshots`` must be created before constructing the adapter.
    The adapter has no retriever callback by design; this makes accidental
    retrieval during candidate evaluation impossible.
    """

    def __init__(self, runnable: Any, retrieval_snapshots: Sequence[RetrievalSnapshot],
                 *, budget: BudgetMetadata, ledger: OptimizationLedger | None = None):
        if not hasattr(runnable, "invoke") or not callable(runnable.invoke):
            raise GepaSetupError("a LangChain Runnable with callable invoke() is required")
        snapshots = tuple(retrieval_snapshots)
        if not snapshots:
            raise ValueError("at least one frozen retrieval snapshot is required")
        by_question = {item.question: item for item in snapshots}
        if len(by_question) != len(snapshots):
            raise ValueError("retrieval snapshot questions must be unique")
        self.runnable = runnable
        self._snapshots = by_question
        self.budget = budget
        self.ledger = ledger or OptimizationLedger()
        self._cache = _Cache()
        self._metric_calls = 0

    @property
    def retrieval_frozen(self) -> bool:
        return True

    @property
    def cache_stats(self) -> Mapping[str, int]:
        return {"hits": self._cache.hits, "misses": self._cache.misses}

    def _case(self, case_or_question: Any) -> RetrievalSnapshot:
        if isinstance(case_or_question, str):
            question = case_or_question
        elif isinstance(case_or_question, Mapping):
            inputs = case_or_question.get("inputs", case_or_question)
            question = inputs.get("question") if isinstance(inputs, Mapping) else None
        else:
            question = None
        if not question or question not in self._snapshots:
            raise KeyError("question is not present in the frozen retrieval snapshot")
        return self._snapshots[question]

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, str):
            return value
        content = getattr(value, "content", None)
        if content is not None:
            return str(content)
        if isinstance(value, Mapping):
            for key in ("answer", "output", "content", "text"):
                if key in value:
                    return str(value[key])
        return str(value)

    def predict(self, prompt: PromptVersion, case_or_question: Any, *, candidate_id: str = "candidate",
                split: str = "train") -> str:
        snapshot = self._case(case_or_question)
        if self._metric_calls >= self.budget.max_metric_calls:
            raise GepaSetupError("GEPA metric-call budget exhausted")
        rendered = prompt.render(question=snapshot.question, context=snapshot.context)
        key = digest({"prompt": prompt.template_digest, "retrieval": snapshot.retrieval_digest, "question": snapshot.question})
        if self.budget.cache_enabled and key in self._cache.values:
            self._cache.hits += 1
            output = self._cache.values[key]
            cache_hit = True
        else:
            self._metric_calls += 1
            self._cache.misses += 1
            # The prompt is explicit so a runnable can either consume it
            # directly or use the question/context fields in a ChatPrompt.
            output = self._content(self.runnable.invoke({
                "question": snapshot.question,
                "context": snapshot.context,
                "prompt": rendered,
            }))
            cache_hit = False
            if self.budget.cache_enabled:
                self._cache.values[key] = output
        result = CandidateResult(
            candidate_id=candidate_id,
            prompt_digest=prompt.template_digest,
            case_id=snapshot.case_id,
            split=split,
            output=output,
            result_digest=digest({"key": key, "output": output}),
            cache_hit=cache_hit,
        )
        self.ledger.append(result)
        return output

    def evaluate(self, prompt: PromptVersion, cases: Sequence[Mapping[str, Any]], *, candidate_id: str,
                 split: str) -> tuple[str, ...]:
        """Evaluate and retain every case result; no score is invented here."""
        return tuple(self.predict(prompt, case, candidate_id=candidate_id, split=split) for case in cases)

    def as_predict_fn(self, prompt: PromptVersion, *, candidate_id: str = "candidate", split: str = "train") -> Callable[[str], str]:
        """Return the callable shape expected by MLflow GEPA."""
        return lambda question: self.predict(prompt, question, candidate_id=candidate_id, split=split)

    def validate_heldout(self, prompt: PromptVersion, cases: Sequence[Mapping[str, Any]],
                         scorer: Callable[[str, Mapping[str, Any]], float], *, candidate_id: str,
                         dataset_digest: str) -> HeldoutValidation:
        """Run the independent validation gate with a caller-supplied real scorer.

        No default score or mock score exists: an empty set, a non-numeric score,
        or a scorer error is an explicit setup/evaluation failure.
        """
        if not cases or not dataset_digest:
            raise GepaSetupError("held-out validation requires a non-empty real dataset digest")
        outputs = self.evaluate(prompt, cases, candidate_id=candidate_id, split="heldout")
        scores: list[float] = []
        for output, case in zip(outputs, cases):
            value = scorer(output, case)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GepaSetupError("held-out scorer must return a numeric score")
            scores.append(float(value))
        return HeldoutValidation(candidate_id, dataset_digest, tuple(scores), len(scores))

    def artifact_links(self, run_id: str) -> tuple[CaseArtifactLink, ...]:
        return self.ledger.links(run_id)

    @property
    def frozen_retrieval_contract(self) -> FrozenRetrievalContract:
        snapshots = tuple(self._snapshots.values())
        return FrozenRetrievalContract(
            snapshot_count=len(snapshots),
            snapshot_digest=digest([snapshot.to_dict() for snapshot in sorted(snapshots, key=lambda item: item.case_id)]),
            case_ids=tuple(snapshot.case_id for snapshot in sorted(snapshots, key=lambda item: item.case_id)),
        )

    def record_selection(self, baseline: HeldoutValidation, selected: HeldoutValidation,
                         *, selection_run_id: str, selection_split: str = "train") -> GepaSelectionRecord:
        """Record selection using observed scores; no score is inferred or filled in."""
        if baseline.dataset_digest != selected.dataset_digest:
            raise GepaSetupError("baseline and selected candidates must use the same dataset digest")
        if selected.candidate_id == baseline.candidate_id:
            raise GepaSetupError("selection must identify a candidate distinct from the baseline")
        prompt_digest = next((item.prompt_digest for item in self.ledger.records
                              if item.candidate_id == selected.candidate_id), "")
        return GepaSelectionRecord(
            selected_candidate_id=selected.candidate_id,
            selected_prompt_digest=prompt_digest,
            baseline_candidate_id=baseline.candidate_id,
            baseline_score=baseline.mean_score,
            selected_score=selected.mean_score,
            dataset_digest=selected.dataset_digest,
            selection_split=selection_split,
            candidate_count=len({item.candidate_id for item in self.ledger.records}),
            retrieval_snapshot_digest=self.frozen_retrieval_contract.snapshot_digest,
            selection_run_id=selection_run_id,
        )

    def log_selection(self, selection: GepaSelectionRecord,
                      *, mlflow_module: Any = None,
                      artifact_path: str = "lineage/gepa_selection.json") -> None:
        if mlflow_module is None:
            try:
                import mlflow as mlflow_module
            except ImportError as exc:
                raise GepaSetupError("MLflow is required to log GEPA selection artifacts") from exc
        mlflow_module.log_dict(selection.to_dict(), artifact_path)
        mlflow_module.set_tags({
            "gepa.selected_candidate_id": selection.selected_candidate_id,
            "gepa.selection_digest": selection.selection_digest,
            "gepa.retrieval_snapshot_digest": selection.retrieval_snapshot_digest,
        })

    def log_ledger(self, *, mlflow_module: Any = None, artifact_path: str = "lineage/candidates.jsonl") -> None:
        if mlflow_module is None:
            try:
                import mlflow as mlflow_module
            except ImportError as exc:
                raise GepaSetupError("MLflow is required to log GEPA candidate artifacts") from exc
        mlflow_module.log_text(self.ledger.jsonl(), artifact_path)
        mlflow_module.log_params({
            "retrieval_frozen": True,
            "retrieval_snapshot_digest": self.frozen_retrieval_contract.snapshot_digest,
            "retrieval_snapshot_count": self.frozen_retrieval_contract.snapshot_count,
            "metric_calls": self._metric_calls,
            "cache_hits": self._cache.hits,
            "cache_misses": self._cache.misses,
        })


def freeze_retrieval(cases: Sequence[Mapping[str, Any]], retrieve: Callable[[str], Any]) -> tuple[RetrievalSnapshot, ...]:
    """Materialize retrieval once per case, before prompt optimization starts.

    This is the only function in this module that accepts a retriever callback.
    Callers should discard it after this function returns.
    """
    snapshots: list[RetrievalSnapshot] = []
    for index, case in enumerate(cases):
        inputs = case.get("inputs", case)
        question = inputs["question"]
        raw = retrieve(question)
        if isinstance(raw, Mapping):
            context = str(raw.get("context", ""))
            document_ids = tuple(str(item) for item in raw.get("document_ids", ()))
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            document_ids = tuple(str(getattr(item, "id_", getattr(item, "id", index))) for item in raw)
            context = "\n\n".join(str(getattr(item, "text", item)) for item in raw)
        else:
            context, document_ids = str(raw), ()
        snapshots.append(RetrievalSnapshot(str(case.get("case_id", f"case-{index}")), question, context, document_ids))
    return tuple(snapshots)


# Short public name for callers that do not need to distinguish the backend.
GepaAdapter = GepaLangChainAdapter
