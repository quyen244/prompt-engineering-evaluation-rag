"""Evaluation runner, modes, per-query output, and paired comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dataset import load_cases, load_manifest, fetch_huggingface_cases, validate_project_cases
from .history import record_document, record_run
from .metrics.retrieval import hit_rate_at_k, mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .retriever import invoke_retriever
from .schema import EvalCase, EvalMode, EvalReport, QueryResult


@dataclass(frozen=True)
class EvalConfig:
    """All retrieval choices are explicit so configurations can be paired."""

    name: str = "unnamed"
    top_k: int = 5
    chunker: str = "external"
    chunk_size: int = 0
    embedding: str = "external"
    retriever: str = "langchain"
    prompt_version: str = "n/a"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "top_k": self.top_k, "chunker": self.chunker,
            "chunk_size": self.chunk_size, "embedding": self.embedding,
            "retriever": self.retriever, "prompt_version": self.prompt_version,
            "metadata": dict(self.metadata),
        }


def mode_cases(mode: EvalMode | str, *, root: str | Path = ".", manifest_path: str | Path | None = None) -> list[EvalCase]:
    selected = EvalMode(mode)
    root_path = Path(root)
    manifest = load_manifest(manifest_path or root_path / "eval_data/manifests/evaluation_manifest.json")
    if selected is EvalMode.SMOKE:
        cases = load_cases(root_path / manifest["project_source"]["path"])
        report = validate_project_cases(cases, root_path)
        if not report.valid:
            raise RuntimeError("project source cases are stale:\n" + "\n".join(report.errors))
    elif selected is EvalMode.IMPROVEMENT:
        cases = fetch_huggingface_cases(manifest, "improvement")
    else:
        cases = fetch_huggingface_cases(manifest, "validation")
    expected = {EvalMode.SMOKE: 20, EvalMode.IMPROVEMENT: 30, EvalMode.VALIDATION: 30}[selected]
    if len(cases) != expected:
        raise RuntimeError(f"{selected.value} requires exactly {expected} real cases, got {len(cases)}")
    return cases


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def run_eval(
    config: EvalConfig,
    cases: Iterable[EvalCase],
    retriever: Any,
    *,
    mode: EvalMode | str = EvalMode.SMOKE,
    output_dir: str | Path = "eval_data/results",
    history_path: str | Path = "eval_data/experiment_history.jsonl",
    run_id: str | None = None,
) -> EvalReport:
    """Run deterministic retrieval scoring against a LangChain retriever.

    No answer generation, synthetic fallback, or fabricated metric is performed.
    A retriever error is allowed to fail the run so an incomplete evaluation is
    never presented as a score.
    """
    selected = EvalMode(mode)
    case_list = list(cases)
    expected = {EvalMode.SMOKE: 20, EvalMode.IMPROVEMENT: 30, EvalMode.VALIDATION: 30}[selected]
    if len(case_list) != expected:
        raise ValueError(f"{selected.value} requires exactly {expected} cases, got {len(case_list)}")
    run_id = run_id or uuid.uuid4().hex
    query_results: list[QueryResult] = []
    for case in case_list:
        retrieved = invoke_retriever(retriever, case.question, config.top_k)
        ids = [doc.document_id for doc in retrieved]
        metrics = {
            "hit_rate": hit_rate_at_k(ids, case.relevant_document_ids, config.top_k),
            "precision": precision_at_k(ids, case.relevant_document_ids, config.top_k),
            "recall": recall_at_k(ids, case.relevant_document_ids, config.top_k),
            "mrr": mrr_at_k(ids, case.relevant_document_ids, config.top_k),
            "ndcg": ndcg_at_k(ids, case.relevant_document_ids, config.top_k),
        }
        for document in retrieved:
            record_document(history_path, run_id=run_id, case_id=case.case_id,
                            document_id=document.document_id, rank=document.rank, score=document.score)
        query_results.append(QueryResult(case.case_id, case.question, retrieved, metrics, run_id=run_id))
    metrics = {name: _mean(result.metrics[name] for result in query_results)
               for name in ("hit_rate", "precision", "recall", "mrr", "ndcg")}
    record_run(history_path, run_id=run_id, mode=selected.value, config=config.to_dict(), metrics=metrics, case_count=len(case_list))
    report = EvalReport(run_id, selected, metrics, tuple(query_results), config.to_dict(), str(history_path))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"{run_id}.jsonl").open("w", encoding="utf-8") as stream:
        for result in query_results:
            stream.write(json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return report


def paired_compare(left: EvalReport, right: EvalReport) -> dict[str, Any]:
    """Compare two configurations on exactly the same case IDs."""
    l = {result.case_id: result for result in left.query_results}
    r = {result.case_id: result for result in right.query_results}
    if set(l) != set(r):
        raise ValueError("paired comparison requires identical case IDs")
    names = ("hit_rate", "precision", "recall", "mrr", "ndcg")
    deltas = {name: sum(r[c].metrics[name] - l[c].metrics[name] for c in l) / len(l) for name in names}
    pair_id = hashlib.sha256((left.run_id + ":" + right.run_id).encode()).hexdigest()[:16]
    return {"pair_id": pair_id, "left_run_id": left.run_id, "right_run_id": right.run_id, "deltas": deltas, "case_count": len(l)}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the source-backed retrieval evaluation harness")
    parser.add_argument("--mode", choices=[mode.value for mode in EvalMode], default=EvalMode.SMOKE.value)
    parser.add_argument("--manifest", default="eval_data/manifests/evaluation_manifest.json")
    parser.parse_args()
    raise SystemExit("A LangChain retriever must be supplied by the application; no implicit/mock retriever is provided.")


if __name__ == "__main__":
    _main()
