"""Controlled TechQA test of document-first prompt layout for prefix reuse.

Uses the existing vLLM server. MLflow artifacts contain hashes and timings, not
question, document, or generated answer text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import uuid
from pathlib import Path

import httpx

from scripts.vllm_techqa_benchmark import (
    CORPUS, MODEL, QUESTIONS, RANKING, SYSTEM, cases, metrics, sha, stream_request,
)

EXPERIMENT = "techqa-vllm-inference"
SECOND_QUESTION = "Please answer this IBM support question from the document: {question}"
LAYOUTS = ("document_first", "question_first")


def make_prompt(context: str, question: str, layout: str) -> str:
    document = f"Documents:\n{context}"
    query = f"Question:\n{question}"
    return f"{document}\n\n{query}" if layout == "document_first" else f"{query}\n\n{document}"


def tokenize(client: httpx.Client, base: str, system: str, prompt: str) -> int:
    response = client.post(base + "/tokenize", json={"model": MODEL, "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": prompt},
    ]})
    response.raise_for_status()
    return response.json()["count"]


def delta(before: dict, after: dict) -> dict:
    return {key: after[key] - before[key] for key in before.keys() & after.keys()
            if key.endswith(("_total", "_sum", "_count"))}


def average(rows: list[dict], key: str) -> float:
    return statistics.mean(row[key] for row in rows if row[key] is not None)


def summarize(rows: list[dict]) -> dict:
    result = {}
    for layout in LAYOUTS:
        own = [row for row in rows if row["layout"] == layout]
        cold = [row for row in own if row["question_index"] == 0]
        followup = [row for row in own if row["question_index"] == 1]
        hits = sum(row["prefix_hits"] for row in own)
        queries = sum(row["prefix_queries"] for row in own)
        result[layout] = {
            "requests": len(own), "errors": sum(bool(row["error"]) for row in own),
            "prompt_tokens_mean": average(own, "prompt_tokens"),
            "output_tokens_mean": average(own, "output_tokens"),
            "cold_ttft_mean_s": average(cold, "ttft_s"),
            "followup_ttft_mean_s": average(followup, "ttft_s"),
            "cold_latency_mean_s": average(cold, "latency_s"),
            "followup_latency_mean_s": average(followup, "latency_s"),
            "prefix_hit_ratio": hits / queries if queries else None,
            "prefix_hits": hits, "prefix_queries": queries,
        }
    a, b = result[LAYOUTS[0]], result[LAYOUTS[1]]
    result["comparison"] = {
        "followup_ttft_reduction_pct": 100 * (1 - a["followup_ttft_mean_s"] / b["followup_ttft_mean_s"]),
        "followup_latency_reduction_pct": 100 * (1 - a["followup_latency_mean_s"] / b["followup_latency_mean_s"]),
        "cold_ttft_reduction_pct": 100 * (1 - a["cold_ttft_mean_s"] / b["cold_ttft_mean_s"]),
        "prompt_token_delta_mean": a["prompt_tokens_mean"] - b["prompt_tokens_mean"],
    }
    pairs = [
        (next(row for row in rows if row["case_id"] == case_id and row["question_index"] == qi
              and row["layout"] == LAYOUTS[0]),
         next(row for row in rows if row["case_id"] == case_id and row["question_index"] == qi
              and row["layout"] == LAYOUTS[1]))
        for case_id in sorted({row["case_id"] for row in rows}) for qi in (0, 1)
    ]
    result["comparison"]["byte_identical_answers"] = sum(
        left["answer_sha256"] == right["answer_sha256"] for left, right in pairs
    )
    result["comparison"]["answer_pairs"] = len(pairs)
    return result


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm_final_optimization"))
    parser.add_argument("--dry-run", action="store_true", help="Only validate exact prompt token counts")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    model_info = httpx.get(base + "/v1/models", timeout=10).json()["data"][0]
    version = httpx.get(base + "/version", timeout=10).json().get("version", "unknown")
    if model_info["id"] != MODEL or model_info.get("max_model_len") != 2048:
        raise RuntimeError(f"Unexpected server model/configuration: {model_info}")
    dataset = cases()
    trial_id = uuid.uuid4().hex
    rows = []
    with httpx.Client(timeout=10) as client:
        for index, case in enumerate(dataset):
            # Counterbalance arm order across documents. Each arm gets a distinct
            # fixed-width nonce at the start of its system message, preventing
            # carryover from the other arm or previous experiments.
            for layout in (LAYOUTS if index % 2 == 0 else LAYOUTS[::-1]):
                arm_marker = "A" if layout == LAYOUTS[0] else "B"
                system = f"{arm_marker}{trial_id}-{index:02d}: {SYSTEM}"
                for question_index, question in enumerate((
                    case["question"], SECOND_QUESTION.format(question=case["question"]),
                )):
                    message = make_prompt(case["context"], question, layout)
                    planned_tokens = tokenize(client, base, system, message)
                    if planned_tokens + 32 > model_info["max_model_len"]:
                        raise RuntimeError(f"Prompt exceeds output budget: {case['id']} {layout}")
                    row = {"case_id": case["id"], "doc_id_sha256": hashlib.sha256(case["doc_id"].encode()).hexdigest(),
                           "layout": layout, "question_index": question_index,
                           "planned_prompt_tokens": planned_tokens,
                           "prompt_sha256": hashlib.sha256(message.encode()).hexdigest()}
                    if not args.dry_run:
                        before = metrics(base)
                        measured = stream_request(base, MODEL, {"id": case["id"], "system": system,
                                                                "prompt": message}, 32, args.timeout)
                        after = metrics(base)
                        counters = delta(before, after)
                        row.update(measured)
                        row["prefix_hits"] = counters.get("prefix_cache_hits_total", 0)
                        row["prefix_queries"] = counters.get("prefix_cache_queries_total", 0)
                        row["server_metric_delta"] = counters
                        print(f"{index+1:02d}/20 {layout} q{question_index} TTFT={row['ttft_s']} error={row['error']}", flush=True)
                    rows.append(row)
    token_diffs = [next(r["planned_prompt_tokens"] for r in rows if r["case_id"] == case["id"]
                        and r["layout"] == LAYOUTS[0] and r["question_index"] == qi)
                   - next(r["planned_prompt_tokens"] for r in rows if r["case_id"] == case["id"]
                          and r["layout"] == LAYOUTS[1] and r["question_index"] == qi)
                   for case in dataset for qi in (0, 1)]
    print(f"A-B prompt token differences: min={min(token_diffs)} max={max(token_diffs)} "
          f"mean={statistics.mean(token_diffs):.3f}", flush=True)
    if args.dry_run:
        return
    if any(row["error"] or row["ttft_s"] is None for row in rows):
        raise RuntimeError("At least one request failed; raw rows were not logged as valid comparison")
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    run_ids = {}
    provenance = {"model": MODEL, "vllm_version": version, "max_model_len": model_info["max_model_len"],
                  "concurrency": 1, "max_tokens": 32, "temperature": 0,
                  "case_count": len(dataset), "requests_per_arm": len(dataset) * 2,
                  "trial_id": trial_id, "source_dense_run_id": "f284f735f18f489bb00cef2b4b417fed",
                  "corpus_sha256": sha(CORPUS), "questions_sha256": sha(QUESTIONS),
                  "ranking_sha256": sha(RANKING), "script_sha256": sha(Path(__file__)),
                  "question_variant": SECOND_QUESTION, "cache_isolation": "nonce_per_doc_arm",
                  "arm_order": "AB_for_even_docs_BA_for_odd_docs", "server_url": base}
    for layout in LAYOUTS:
        own = [row for row in rows if row["layout"] == layout]
        with mlflow.start_run(run_name=f"final-optimization-{layout}") as run:
            run_ids[layout] = run.info.run_id
            mlflow.log_params({**provenance, "layout": layout})
            mlflow.log_metrics({key: value for key, value in summary[layout].items()
                                if isinstance(value, (int, float)) and value is not None})
            mlflow.log_dict({"trial_id": trial_id, "layout": layout, "rows": own,
                             "summary": summary[layout], "token_diffs": token_diffs},
                            "measurements/record.json")
    record = {"trial_id": trial_id, "run_ids": run_ids, "provenance": provenance,
              "token_diffs": token_diffs, "summary": summary, "rows": rows}
    output = args.output / f"{trial_id}.json"
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"run_ids": run_ids, "artifact": str(output), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
