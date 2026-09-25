"""Run paired retrieval variants with a caller-supplied retriever factory."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

from src.eval.dataset import load_manifest
from src.eval.runner import EvalConfig, mode_cases, paired_compare, run_eval
from src.eval.schema import EvalCase, EvalReport


RetrieverFactory = Callable[[EvalConfig, dict[str, Any]], Any]


def build_configs(top_k_values: Sequence[int], embedding_values: Sequence[str], retriever: str,
                  metadata: dict[str, Any] | None = None) -> list[EvalConfig]:
    """Build the Cartesian experiment matrix while keeping all other choices fixed."""
    if not top_k_values or not embedding_values:
        raise ValueError("at least one Top-K value and embedding are required")
    if len(set(top_k_values)) != len(top_k_values):
        raise ValueError("Top-K values must be unique")
    if len(set(embedding_values)) != len(embedding_values):
        raise ValueError("embedding values must be unique")
    return [EvalConfig(
        name=f"{embedding.replace(':', '-')}-{retriever}-topk-{top_k}",
        top_k=top_k,
        embedding=embedding,
        retriever=retriever,
        metadata=metadata or {},
    ) for embedding in embedding_values for top_k in top_k_values]


def _load_factory(spec: str) -> RetrieverFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use module.path:callable format")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"retriever factory is not callable: {spec}")
    return factory


def _dataset_fingerprint(cases: Sequence[EvalCase]) -> str:
    stable_rows = [
        {"case_id": case.case_id, "row_id": case.row_id, "row_hash": case.row_hash,
         "source": case.source, "source_spans": [span.to_dict() for span in case.source_spans]}
        for case in cases
    ]
    payload = json.dumps(stable_rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_ablation(
    *,
    cases: Sequence[EvalCase],
    configs: Sequence[EvalConfig],
    retriever_factory: RetrieverFactory,
    mlflow_module: Any,
    mode: str,
    experiment_name: str,
    tracking_uri: str,
    factory_options: dict[str, Any] | None = None,
    output_dir: str | Path = "eval_data/results",
    history_path: str | Path = "eval_data/experiment_history.jsonl",
    manifest_path: str | Path = "eval_data/manifests/evaluation_manifest.json",
) -> tuple[str, list[EvalReport], dict[str, Any]]:
    """Execute each variant with injected data, retrieval, and MLflow dependencies."""
    if not cases or not configs:
        raise ValueError("at least one evaluation case and config are required")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation cases must have unique IDs")

    mlflow_module.set_tracking_uri(tracking_uri)
    mlflow_module.set_experiment(experiment_name)
    study_id = uuid.uuid4().hex
    dataset_hash = _dataset_fingerprint(cases)
    options = factory_options or {}
    reports: list[EvalReport] = []
    mlflow_run_ids: list[str] = []
    manifest = load_manifest(manifest_path)
    manifest_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    for config in configs:
        retriever = retriever_factory(config, options)
        if retriever is None:
            raise RuntimeError(f"retriever factory returned None for {config.name}")
        with mlflow_module.start_run(run_name=config.name) as active_run:
            mlflow_run_id = active_run.info.run_id
            mlflow_run_ids.append(mlflow_run_id)
            mlflow_module.set_tags({
                "study_id": study_id,
                "run_kind": "retrieval_ablation",
                "eval_mode": mode,
                "comparison_design": "paired",
                "dataset_fingerprint": dataset_hash,
                "manifest_digest": manifest_digest,
            })
            mlflow_module.log_params({
                "experiment_name": config.name,
                "top_k": config.top_k,
                "chunker": config.chunker,
                "chunk_size": config.chunk_size,
                "embedding": config.embedding,
                "retriever": config.retriever,
                "prompt_version": config.prompt_version,
                "case_count": len(cases),
                "factory_options": json.dumps(options, sort_keys=True),
                **{f"config.{key}": str(value) for key, value in config.metadata.items()},
            })
            report = run_eval(
                config, cases, retriever,
                mode=mode,
                output_dir=output_dir,
                history_path=history_path,
                run_id=mlflow_run_id,
            )
            reports.append(report)
            mlflow_module.log_metrics(dict(report.metrics))
            mlflow_module.log_dict({
                "study_id": study_id,
                "run_id": mlflow_run_id,
                "dataset_fingerprint": dataset_hash,
                "manifest_digest": manifest_digest,
                "config": config.to_dict(),
                "case_count": len(cases),
                "metrics": dict(report.metrics),
            }, "evaluation/summary.json")
            mlflow_module.log_dict({
                "run_id": mlflow_run_id,
                "results": [
                    {"case_id": result.case_id, "retrieved": [
                        {"document_id": item.document_id, "rank": item.rank, "score": item.score}
                        for item in result.retrieved
                    ], "metrics": dict(result.metrics)}
                    for result in report.query_results
                ],
            }, "evaluation/per_case.json")
            mlflow_module.log_dict(manifest, "provenance/evaluation_manifest.json")

    deltas = []
    for report in reports[1:]:
        deltas.append(paired_compare(reports[0], report))
    comparison = {
        "study_id": study_id,
        "baseline_run_id": mlflow_run_ids[0],
        "run_ids": mlflow_run_ids,
        "dataset_fingerprint": dataset_hash,
        "case_count": len(cases),
        "paired_deltas": deltas,
    }
    with mlflow_module.start_run(run_id=mlflow_run_ids[0]):
        mlflow_module.log_dict(comparison, "comparison/paired_deltas.json")
    return study_id, reports, comparison


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired retrieval ablations; inject your retriever as module:factory."
    )
    parser.add_argument(
        "--factory", default="scripts.project_bm25_factory:build_retriever",
        help="Retriever factory: package.module:callable(config, options)",
    )
    parser.add_argument("--mode", choices=("smoke", "improvement", "validation"), default="smoke")
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--embedding", nargs="+", default=["none"],
                        help="One or more embedding IDs; runs every embedding x Top-K combination")
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("--experiment", default="rag-retrieval-ablation")
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--manifest", default="eval_data/manifests/evaluation_manifest.json")
    parser.add_argument("--factory-options", default="{}", help="JSON object passed to the injected factory")
    parser.add_argument("--output-dir", default="eval_data/results")
    parser.add_argument("--history-path", default="eval_data/experiment_history.jsonl")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        options = json.loads(args.factory_options)
        if not isinstance(options, dict):
            raise ValueError("factory-options must be a JSON object")
        import mlflow
    except (ImportError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(f"Cannot start ablation: {error}") from error

    factory = _load_factory(args.factory)
    cases = mode_cases(args.mode, root=".", manifest_path=args.manifest)
    try:
        configs = build_configs(args.top_k, args.embedding, args.retriever, options)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    study_id, reports, comparison = run_ablation(
        cases=cases,
        configs=configs,
        retriever_factory=factory,
        mlflow_module=mlflow,
        mode=args.mode,
        experiment_name=args.experiment,
        tracking_uri=args.tracking_uri,
        factory_options=options,
        output_dir=args.output_dir,
        history_path=args.history_path,
        manifest_path=args.manifest,
    )
    print(f"Study: {study_id}")
    for report in reports:
        scores = " ".join(f"{name}={value:.4f}" for name, value in report.metrics.items())
        print(f"{report.run_id} {report.config['name']} {scores}")
    print(f"Paired deltas: {json.dumps(comparison['paired_deltas'], sort_keys=True)}")


if __name__ == "__main__":
    main()
