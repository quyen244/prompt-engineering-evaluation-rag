"""Evaluate the frozen dense TechQA retrieval with vLLM and RAGAS.

Only hashes, scores, usage, and provenance are persisted.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import httpx

from scripts.vllm_techqa_benchmark import CORPUS, MODEL, QUESTIONS, RANKING, SYSTEM, cases, sha
from scripts.techqa_generation import _source_context
from scripts.techqa_retrieval import CHUNK_OVERLAP, CHUNK_SIZE, load_dataset

JUDGE = "liquid/lfm-2.5-2.6b:free"


def generate(base: str, case: dict, max_tokens: int) -> tuple[str, dict, float]:
    prompt = f"Documents:\n{case['context']}\n\nQuestion:\n{case['question']}"
    start = time.perf_counter()
    response = httpx.post(base.rstrip("/") + "/v1/chat/completions", json={
        "model": MODEL, "messages": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": max_tokens,
    }, timeout=120)
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("vLLM returned empty answer")
    return answer.strip(), data.get("usage") or {}, time.perf_counter() - start


def main() -> None:
    import mlflow
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness, FactualCorrectness, ContextRecall

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--judge-model", default=JUDGE)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--case-id", help="Evaluate one TechQA test case for a targeted retry")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--ranking", type=Path, default=RANKING)
    parser.add_argument("--retrieval-run-id", default="f284f735f18f489bb00cef2b4b417fed")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm_benchmark/ragas_scores.json"))
    args = parser.parse_args()
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    info = httpx.get(args.base_url.rstrip("/") + "/v1/models", timeout=10).json()["data"][0]
    if info["id"] != MODEL or info.get("max_model_len", 0) < 1024:
        raise RuntimeError("Unexpected vLLM model or context limit")
    client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                         timeout=120, max_retries=2)
    judge = llm_factory(args.judge_model, client=client, max_tokens=args.judge_max_tokens, temperature=0)
    faithfulness = Faithfulness(llm=judge)
    factual = FactualCorrectness(llm=judge)
    recall = ContextRecall(llm=judge)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("techqa-vllm-ragas")
    documents = {doc["doc_id"]: doc for doc in load_dataset(CORPUS, QUESTIONS)[0]}
    evaluation_cases = []
    selected_cases = [case for case in cases(args.ranking) if not args.case_id or case["id"] == args.case_id]
    if args.case_id and not selected_cases:
        raise ValueError(f"Unknown case ID: {args.case_id}")
    for case in selected_cases[:args.limit]:
        doc = dict(documents[case["doc_id"]])
        index = int(case["chunk_id"].rsplit("-", 1)[1])
        start = index * (CHUNK_SIZE - CHUNK_OVERLAP)
        doc["chunk_id"] = case["chunk_id"]
        doc["context_text"] = f"{doc['title']}\n\n{doc['text'][start:start + CHUNK_SIZE]}"
        context, provenance = _source_context([doc], case["question"], 3500)
        evaluation_cases.append({**case, "context": context, "context_provenance": provenance})
    for case in evaluation_cases:
        prompt = f"Documents:\n{case['context']}\n\nQuestion:\n{case['question']}"
        tokenized = httpx.post(args.base_url.rstrip("/") + "/tokenize",
                               json={"model": MODEL, "prompt": prompt}, timeout=20)
        tokenized.raise_for_status()
        case["prompt_tokens_preflight"] = tokenized.json()["count"]
        if case["prompt_tokens_preflight"] + args.max_tokens + 48 > info["max_model_len"]:
            raise RuntimeError(f"Prompt exceeds vLLM context limit for {case['id']}")
    if args.validate_only:
        print(json.dumps({"cases": len(evaluation_cases),
                          "max_prompt_tokens": max(x["prompt_tokens_preflight"] for x in evaluation_cases),
                          "max_model_len": info["max_model_len"]}))
        return
    rows = []
    with mlflow.start_run(run_name="dense-top1-vllm-ragas") as run:
        mlflow.log_params({"generator_model": MODEL, "judge_model": args.judge_model,
                           "ragas_version": __import__("ragas").__version__,
                           "test_count": len(evaluation_cases), "max_tokens": args.max_tokens,
                           "judge_max_tokens": args.judge_max_tokens,
                           "max_model_len": info.get("max_model_len"),
                           "retrieval_method": "dense", "retrieval_top_k": 1,
                           "embedding": "ollama:qwen3-embedding:0.6b",
                           "frozen_retrieval_run_id": args.retrieval_run_id,
                           "ranking_sha256": sha(args.ranking), "corpus_sha256": sha(CORPUS),
                           "questions_sha256": sha(QUESTIONS),
                           "context_policy": "retrieved_chunk_first_question_windows_v2",
                           "max_context_chars": 3500,
                           "chunk_size_chars": 850, "chunk_overlap_chars": 150})
        for index, case in enumerate(evaluation_cases, 1):
            print(f"evaluating {index}/{len(evaluation_cases)}: {case['id']}", flush=True)
            answer, usage, generation_s = generate(args.base_url, case, args.max_tokens)
            start = time.perf_counter()
            error = None
            try:
                async def score_case() -> tuple[dict[str, float], list[str]]:
                    values = {}
                    errors = []
                    names = ("faithfulness", "factual_correctness", "context_recall")
                    results = await asyncio.gather(
                        faithfulness.ascore(user_input=case["question"], response=answer,
                                           retrieved_contexts=[case["context"]]),
                        factual.ascore(response=answer, reference=case["reference"]),
                        recall.ascore(user_input=case["question"], retrieved_contexts=[case["context"]],
                                      reference=case["reference"]),
                        return_exceptions=True,
                    )
                    for name, result in zip(names, results):
                        if isinstance(result, BaseException):
                            errors.append(f"{name}: {type(result).__name__}: {str(result)[:120]}")
                            continue
                        value = float(result.value)
                        if math.isfinite(value):
                            values[name] = value
                    return values, errors
                scores, metric_errors = asyncio.run(score_case())
                error = "; ".join(metric_errors) if metric_errors else None
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:180]}"
                scores = {}
            row = {"case_id": case["id"], "doc_id": case["doc_id"],
                   "chunk_id": case["chunk_id"], "context_chars": len(case["context"]),
                   "context_provenance": case["context_provenance"],
                   "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                   "answer_chars": len(answer), "prompt_tokens": usage.get("prompt_tokens"),
                   "prompt_tokens_preflight": case["prompt_tokens_preflight"],
                   "completion_tokens": usage.get("completion_tokens"),
                   "generation_s": generation_s, "ragas_s": time.perf_counter() - start,
                   "scores": scores, "error": error}
            rows.append(row)
            mlflow.log_dict(row, f"cases/{index:02d}.json")
            print(json.dumps({"case_id": case["id"], "scores": scores, "error": error}), flush=True)
        aggregate = {name: statistics.mean(row["scores"][name] for row in rows if name in row["scores"])
                     for name in ("faithfulness", "factual_correctness", "context_recall")
                     if any(name in row["scores"] for row in rows)}
        aggregate.update({"scored_cases": sum(bool(row["scores"]) for row in rows),
                          "failed_cases": sum(bool(row["error"]) for row in rows),
                          "generation_s_mean": statistics.mean(row["generation_s"] for row in rows)})
        mlflow.log_metrics(aggregate)
        report = {"run_id": run.info.run_id, "model": MODEL, "judge_model": args.judge_model,
                  "dataset_count": len(rows), "aggregate": aggregate, "rows": rows}
        mlflow.log_dict(report, "evaluation/summary.json")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_id": run.info.run_id, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
