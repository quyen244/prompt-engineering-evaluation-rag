"""Immutable experiment lineage primitives for RAG and prompt optimization.

This module deliberately contains no MLflow or retriever implementation.  It is
safe to use in an offline validation job and can be imported when optional
tracking dependencies are not installed.
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping, Sequence


class LineageError(ValueError):
    """Base error for invalid lineage metadata."""


class PromptVariableError(LineageError):
    """Raised when a prompt has missing or unresolved variables."""


class PromotionRefused(LineageError):
    """Raised when a candidate does not have enough evidence to be promoted."""


def _plain(value: Any) -> Any:
    """Convert metadata to JSON-safe, deterministic values."""
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _plain(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list, frozenset, set)):
        values = [_plain(v) for v in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True)) if isinstance(value, (frozenset, set)) else values
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    """Return a content identity, never an identity based on a mutable alias."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_VARIABLE_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")


def prompt_variables(template: str) -> frozenset[str]:
    if not isinstance(template, str) or not template.strip():
        raise PromptVariableError("prompt template must be a non-empty string")
    return frozenset(_VARIABLE_RE.findall(template))


def validate_prompt_variables(template: str, required: Sequence[str] | None = None) -> frozenset[str]:
    found = prompt_variables(template)
    if required is not None:
        expected = frozenset(required)
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing variables: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected variables: {', '.join(unexpected)}")
            raise PromptVariableError("; ".join(details))
    return found


@dataclass(frozen=True)
class PromptVersion:
    """An immutable prompt version.  Aliases intentionally do not live here."""

    name: str
    version: int
    template: str
    variables: frozenset[str] = field(default_factory=frozenset)
    template_digest: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.version < 1:
            raise LineageError("prompt name and positive version are required")
        found = prompt_variables(self.template)
        supplied = frozenset(self.variables)
        if supplied and supplied != found:
            raise PromptVariableError(f"declared variables {sorted(supplied)} do not match template {sorted(found)}")
        object.__setattr__(self, "variables", found)
        expected_digest = digest(self.template)
        if self.template_digest and self.template_digest != expected_digest:
            raise LineageError("prompt digest does not match its immutable template")
        object.__setattr__(self, "template_digest", expected_digest)

    @property
    def uri(self) -> str:
        return f"prompts:/{self.name}/{self.version}"

    @property
    def digest(self) -> str:
        return self.template_digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "template_digest": self.template_digest,
            "variables": sorted(self.variables),
        }

    def render(self, **values: Any) -> str:
        missing = sorted(self.variables - values.keys())
        if missing:
            raise PromptVariableError(f"missing prompt variables: {', '.join(missing)}")
        rendered = self.template
        for variable in self.variables:
            rendered = re.sub(r"{{\s*" + re.escape(variable) + r"\s*}}", str(values[variable]), rendered)
        if _VARIABLE_RE.search(rendered):
            raise PromptVariableError("prompt contains unresolved variables")
        return rendered


# The name used by the phase-2 documentation is retained as a public alias.
PromptRef = PromptVersion


@dataclass(frozen=True)
class PromptAlias:
    """A movable label pointing at an immutable prompt version."""

    name: str
    alias: str
    version: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "alias": self.alias, "version": self.version}


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    name: str
    version: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(_plain(self.parameters))))

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "name": self.name, "version": self.version, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class DatasetIdentity:
    name: str
    dataset_digest: str
    split: str
    case_count: int

    def __post_init__(self) -> None:
        if self.case_count < 1 or not self.dataset_digest:
            raise LineageError("a real, non-empty dataset digest and case count are required")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dataset_digest": self.dataset_digest, "split": self.split, "case_count": self.case_count}


@dataclass(frozen=True)
class RetrievalIdentity:
    """Identity for the retrieval configuration, not its implementation."""

    strategy: str
    top_k: int
    bm25_digest: str = ""
    reranker_digest: str = ""
    dense_digest: str = ""
    hybrid_digest: str = ""

    def __post_init__(self) -> None:
        if not self.strategy or self.top_k < 1:
            raise LineageError("retrieval strategy and positive top_k are required")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class BudgetMetadata:
    max_metric_calls: int
    max_reflection_calls: int = 0
    max_cost_usd: float | None = None
    cache_enabled: bool = False
    cache_namespace: str = ""

    def __post_init__(self) -> None:
        if self.max_metric_calls < 1 or self.max_reflection_calls < 0:
            raise LineageError("optimization budgets must be non-negative and metric budget must be positive")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise LineageError("max_cost_usd must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class RollbackMetadata:
    previous_prompt_version: int
    previous_alias: str
    rollback_run_id: str
    reason: str

    def __post_init__(self) -> None:
        if self.previous_prompt_version < 1 or not self.previous_alias or not self.rollback_run_id or not self.reason:
            raise LineageError("rollback metadata must identify the previous version, alias, run, and reason")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ExperimentManifest:
    """Content-addressed manifest tying one document experiment together."""

    retrieval: RetrievalIdentity
    prompt: PromptVersion
    model: ModelIdentity
    dataset: DatasetIdentity
    budget: BudgetMetadata
    git_commit: str = ""
    schema_version: str = "1"
    rollback: RollbackMetadata | None = None
    manifest_digest: str = ""

    def __post_init__(self) -> None:
        payload = self.to_dict(include_digest=False)
        expected_digest = digest(payload)
        if self.manifest_digest and self.manifest_digest != expected_digest:
            raise LineageError("manifest digest does not match its immutable contents")
        object.__setattr__(self, "manifest_digest", expected_digest)

    @property
    def identity(self) -> str:
        return self.manifest_digest

    def to_dict(self, include_digest: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "retrieval": self.retrieval.to_dict(),
            "prompt": self.prompt.to_dict(),
            "model": self.model.to_dict(),
            "dataset": self.dataset.to_dict(),
            "budget": self.budget.to_dict(),
            "rollback": self.rollback.to_dict() if self.rollback else None,
        }
        if include_digest:
            data["manifest_digest"] = self.manifest_digest
        return data


def build_manifest(*, retrieval: RetrievalIdentity, prompt: PromptVersion, model: ModelIdentity,
                   dataset: DatasetIdentity, budget: BudgetMetadata,
                   git_commit: str = "", rollback: RollbackMetadata | None = None,
                   schema_version: str = "1") -> ExperimentManifest:
    """Build a stable manifest; changing any component changes its identity."""
    return ExperimentManifest(retrieval, prompt, model, dataset, budget, git_commit, schema_version, rollback)


@dataclass(frozen=True)
class RunIdentity:
    experiment_id: str
    run_id: str
    kind: str
    parent_run_id: str = ""
    candidate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def parent_run_name(manifest: ExperimentManifest) -> str:
    return f"rag-experiment-{manifest.identity[:12]}"


def child_run_name(candidate_id: str, split: str) -> str:
    return f"candidate-{candidate_id}-{split}"


@contextmanager
def start_parent_run(manifest: ExperimentManifest, *, experiment: str = "rag-lineage"):
    """Start the one parent run used for a manifest and yield its identity."""
    try:
        import mlflow
    except ImportError as exc:
        raise LineageError("MLflow is required for tracked runs; install requirements/base.txt") from exc
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=parent_run_name(manifest)) as run:
        log_manifest(manifest)
        mlflow.set_tags({"run_kind": "parent", "manifest_digest": manifest.identity})
        yield RunIdentity(experiment, run.info.run_id, "parent")


@contextmanager
def start_child_run(parent: RunIdentity, *, candidate_id: str, split: str):
    """Start a nested candidate/validation run under the parent run."""
    try:
        import mlflow
    except ImportError as exc:
        raise LineageError("MLflow is required for tracked runs; install requirements/base.txt") from exc
    with mlflow.start_run(run_name=child_run_name(candidate_id, split), nested=True) as run:
        mlflow.set_tags({"run_kind": "child", "parent_run_id": parent.run_id,
                         "candidate_id": candidate_id, "split": split})
        yield RunIdentity(parent.experiment_id, run.info.run_id, "child", parent.run_id, candidate_id)


@dataclass(frozen=True)
class CaseArtifactLink:
    case_id: str
    split: str
    run_id: str
    artifact_path: str
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class PromotionEvidence:
    baseline_score: float | None
    candidate_score: float | None
    heldout_case_count: int
    heldout_evaluated: bool
    retrieval_frozen: bool
    candidates_recorded: bool
    real_data: bool
    dataset_digest: str
    candidate_run_id: str
    per_case_artifacts: tuple[CaseArtifactLink, ...] = ()

    def missing(self) -> tuple[str, ...]:
        checks = {
            "baseline_score": self.baseline_score is not None,
            "candidate_score": self.candidate_score is not None,
            "heldout_evaluation": self.heldout_evaluated and self.heldout_case_count > 0,
            "frozen_retrieval": self.retrieval_frozen,
            "candidate_records": self.candidates_recorded,
            "real_dataset": self.real_data and bool(self.dataset_digest),
            "candidate_run_id": bool(self.candidate_run_id),
            "per_case_artifacts": bool(self.per_case_artifacts),
        }
        return tuple(name for name, present in checks.items() if not present)

    @property
    def improvement_observed(self) -> bool:
        return self.baseline_score is not None and self.candidate_score is not None and self.candidate_score > self.baseline_score


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...] = ()


def assess_promotion(evidence: PromotionEvidence) -> PromotionDecision:
    missing = evidence.missing()
    reasons = tuple(f"incomplete evidence: {item}" for item in missing)
    if not evidence.improvement_observed:
        reasons += ("candidate does not demonstrate improvement on held-out real data",)
    return PromotionDecision(not reasons, reasons)


def require_promotion(evidence: PromotionEvidence) -> PromotionDecision:
    decision = assess_promotion(evidence)
    if not decision.eligible:
        raise PromotionRefused("; ".join(decision.reasons))
    return decision


def log_case_artifact_links(links: Sequence[CaseArtifactLink], *, artifact_path: str = "lineage/case_links.json") -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise LineageError("MLflow is required to log case artifact links") from exc
    mlflow.log_dict({"cases": [link.to_dict() for link in links]}, artifact_path)


def log_manifest(manifest: ExperimentManifest, *, artifact_path: str = "lineage/manifest.json") -> None:
    """Log an immutable manifest to the active MLflow run.

    MLflow is intentionally imported only here so offline manifest validation
    remains usable when optional tracking dependencies are absent.
    """
    try:
        import mlflow
    except ImportError as exc:  # explicit setup failure, never a silent no-op
        raise LineageError("MLflow is required to log experiment lineage; install requirements/base.txt") from exc
    mlflow.log_dict(manifest.to_dict(), artifact_path)
    mlflow.set_tag("manifest_digest", manifest.identity)
