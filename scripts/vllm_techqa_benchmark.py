"""Measure the existing vLLM server on the frozen TechQA test set.

No question, context, answer, or credential text is written to MLflow.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx

from scripts.techqa_retrieval import CHUNK_OVERLAP, CHUNK_SIZE, load_dataset

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = "Answer only from the supplied IBM document. If evidence is insufficient, say so. Be concise."
RANKING = Path("mlruns/8/f284f735f18f489bb00cef2b4b417fed/artifacts/evaluation/per_case.json")
CORPUS = Path("eval_data/rrf_study/corpus.jsonl")
QUESTIONS = Path("eval_data/rrf_study/test_questions.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases(ranking_path: Path = RANKING) -> list[dict]:
    docs, questions, _ = load_dataset(CORPUS, QUESTIONS)
    doc_map = {d["doc_id"]: d for d in docs}
    ranking_artifact = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranking = ranking_artifact.get("rows", ranking_artifact.get("results"))
    if not isinstance(ranking, list):
        raise ValueError("Dense ranking artifact must contain rows or results")
    ranks = {r["question_id"]: r for r in ranking}
    if len(questions) != len(ranks):
        raise ValueError("Frozen dense ranking does not match test question count")
    result = []
    for q in questions:
        r = ranks[q["id"]]
        doc_id = r.get("retrieved_doc_ids_at_5", r.get("retrieved_doc_ids"))[0]
        chunk_id = r.get("retrieved_chunk_ids_at_5", r.get("retrieved_chunk_ids"))[0]
        if doc_id != q["gold_doc_id"] or not chunk_id.startswith(doc_id + "::chunk-"):
            raise ValueError(f"Frozen dense top-1 is inconsistent for {q['id']}")
        d = doc_map[doc_id]
        index = int(chunk_id.rsplit("-", 1)[1])
        start = index * (CHUNK_SIZE - CHUNK_OVERLAP)
        chunk = d["text"][start:start + CHUNK_SIZE]
        result.append({"id": q["id"], "question": q["question"],
                       "reference": q["ground_truth"], "doc_id": doc_id,
                       "chunk_id": chunk_id, "context": f"[doc:{doc_id}]\n{d['title']}\n\n{chunk}"})
    return result


def metrics(base: str) -> dict[str, float]:
    try:
        text = httpx.get(base.rstrip("/") + "/metrics", timeout=10).text
    except httpx.HTTPError:
        return {}
    names = ("prefix_cache_queries_total", "prefix_cache_hits_total", "request_queue_time_seconds_sum",
             "request_queue_time_seconds_count", "request_prefill_time_seconds_sum",
             "request_prefill_time_seconds_count", "request_num_preemptions_sum", "kv_cache_usage_perc")
    values = {}
    for name in names:
        match = re.search(r"^vllm:" + re.escape(name) + r"(?:\{[^\n]*?\})?\s+([0-9.eE+-]+)$", text, re.M)
        if match:
            values[name] = float(match.group(1))
    return values


def gpu_stats() -> tuple[float | None, float | None]:
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                                         timeout=5, stderr=subprocess.DEVNULL).decode().strip().splitlines()[0]
        memory, temperature = output.split(",")
        return float(memory), float(temperature)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None, None


def stream_request(base: str, model: str, item: dict, max_tokens: int, timeout: float) -> dict:
    started = time.perf_counter()
    first = None
    usage = {}
    chunks = []
    error = None
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", base.rstrip("/") + "/v1/chat/completions", json={
                "model": model, "messages": [{"role": "system", "content": item.get("system", SYSTEM)},
                                                {"role": "user", "content": item["prompt"]}],
                "temperature": 0, "max_tokens": item.get("max_tokens", max_tokens), "stream": True,
                "stream_options": {"include_usage": True},
            }) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[6:])
                    usage = data.get("usage") or usage
                    for choice in data.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content:
                            if first is None:
                                first = time.perf_counter()
                            chunks.append(content)
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:150]}"
    finished = time.perf_counter()
    output_tokens = usage.get("completion_tokens") or 0
    latency = finished - started
    return {"case_id": item["id"], "input_variant": item.get("variant", "rag"),
            "prompt_tokens": usage.get("prompt_tokens"), "output_tokens": output_tokens,
            "ttft_s": first - started if first else None, "latency_s": latency,
            "tpot_s": (finished - first) / (output_tokens - 1) if first and output_tokens > 1 else None,
            "output_tps": output_tokens / latency if output_tokens else None,
            "answer_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest() if chunks else None,
            "error": error}


def prompt(item: dict, target_chars: int | None = None, variant: str = "rag") -> dict:
    context = item["context"]
    if target_chars:
        context = (context + "\n") * math.ceil(target_chars / (len(context) + 1))
        context = context[:target_chars]
    return {"id": item["id"], "variant": variant,
            "prompt": f"Documents:\n{context}\n\nQuestion:\n{item['question']}"}


def run_batch(base: str, model: str, items: list[dict], concurrency: int, max_tokens: int,
              timeout: float) -> tuple[list[dict], dict]:
    before = metrics(base)
    peak = [gpu_stats()]
    stop = threading.Event()
    def sample_gpu():
        while not stop.wait(0.3):
            peak.append(gpu_stats())
    sampler = threading.Thread(target=sample_gpu, daemon=True)
    sampler.start()
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(lambda item: stream_request(base, model, item, max_tokens, timeout), items))
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=2)
    after = metrics(base)
    values = [x[0] for x in peak if x[0] is not None]
    temperatures = [x[1] for x in peak if x[1] is not None]
    delta = {name: after[name] - before[name] for name in before.keys() & after.keys()
             if name.endswith(("_total", "_sum", "_count"))}
    return rows, {"wall_s": elapsed, "gpu_peak_mib": max(values) if values else None,
                  "gpu_start_mib": values[0] if values else None,
                  "gpu_peak_temp_c": max(temperatures) if temperatures else None,
                  "server_metric_delta": delta}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    left = int(position)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)


def summary(rows: list[dict], details: dict) -> dict:
    good = [r for r in rows if not r["error"]]
    lat = [r["latency_s"] for r in good]
    ttft = [r["ttft_s"] for r in good if r["ttft_s"] is not None]
    output = sum(r["output_tokens"] for r in good)
    delta = details["server_metric_delta"]
    queries = delta.get("prefix_cache_queries_total", 0)
    return {"requests": len(rows), "success": len(good), "errors": len(rows) - len(good),
            "latency_mean_s": statistics.mean(lat) if lat else None,
            "latency_p95_s": percentile(lat, .95), "latency_p99_s": percentile(lat, .99) if len(lat) >= 20 else None,
            "ttft_mean_s": statistics.mean(ttft) if ttft else None,
            "ttft_p95_s": percentile(ttft, .95),
            "request_tps": len(good) / details["wall_s"], "output_tps": output / details["wall_s"],
            "gpu_peak_mib": details["gpu_peak_mib"],
            "gpu_peak_temp_c": details.get("gpu_peak_temp_c"),
            "prefix_hit_ratio": delta.get("prefix_cache_hits_total", 0) / queries if queries else None,
            "server_queue_mean_s": (delta["request_queue_time_seconds_sum"] / delta["request_queue_time_seconds_count"]
                                    if delta.get("request_queue_time_seconds_count", 0) else None),
            "server_prefill_mean_s": (delta["request_prefill_time_seconds_sum"] / delta["request_prefill_time_seconds_count"]
                                      if delta.get("request_prefill_time_seconds_count", 0) else None),
            "server_metric_delta": delta}


def scenarios(data: list[dict]):
    base = [prompt(x) for x in data]
    for size, tokens in ((300, 32), (1300, 32), (4500, 32), (1300, 128)):
        nonce = uuid.uuid4().hex
        items = [{**prompt(x, size), "system": f"{SYSTEM} Baseline trial {nonce}-{i}."}
                 for i, x in enumerate(data[:3])]
        yield "baseline", f"chars-{size}-out-{tokens}", items, 1, tokens
    for concurrency in (1, 2, 4, 8, 16, 32):
        count = max(8, concurrency)
        nonce = uuid.uuid4().hex
        items = [{**base[i % len(base)], "id": f"{base[i % len(base)]['id']}-{i}",
                  "system": f"{SYSTEM} Concurrency trial {nonce}-{i}."} for i in range(count)]
        yield "concurrency", f"c-{concurrency}", items, concurrency, 64
    shared = data[0]["context"] * 4
    for variant in ("unique", "system_shared", "doc_shared", "partial_shared", "reordered"):
        salt = uuid.uuid4().hex
        items = []
        for index, case in enumerate(data[:6]):
            if variant == "unique":
                context = case["context"] * 3
            elif variant == "system_shared":
                context = case["context"] * 3
            elif variant == "doc_shared":
                context = shared
            elif variant == "partial_shared":
                context = shared[:len(shared)//2] + case["context"]
            else:
                parts = shared.split("\n")
                context = "\n".join(reversed(parts)) if index % 2 else shared
            items.append({"id": case["id"], "variant": variant,
                          "system": (f"{SYSTEM} Cache experiment {salt}-{case['id']}." if variant == "unique"
                                     else f"{SYSTEM} Cache experiment {salt}."),
                          "prompt": f"Documents:\n{context}\nQuestion:\n{case['question']}"})
        yield "prefix", variant + "-cold", items, 1, 32
        yield "prefix", variant + "-warm", items, 1, 32
    for size in (700, 1800, 3500, 6000):
        for concurrency in (1, 4):
            nonce = uuid.uuid4().hex
            items = [{**prompt(x, size), "system": f"{SYSTEM} Context trial {nonce}-{i}."}
                     for i, x in enumerate(data[:concurrency*2])]
            yield "context", f"chars-{size}-c-{concurrency}", items, concurrency, 32
    variants = [(400, 32), (400, 128), (4500, 32), (4500, 128)]
    for index, (size, tokens) in enumerate(variants):
        nonce = uuid.uuid4().hex
        items = [{**prompt(x, size), "system": f"{SYSTEM} Composition trial {nonce}-{i}."}
                 for i, x in enumerate(data[:8])]
        yield "composition", f"type-{index}", items, 4, tokens
    nonce = uuid.uuid4().hex
    mixed = [{**prompt(case, variants[i % 4][0], f"type-{i % 4}"),
              "system": f"{SYSTEM} Composition trial {nonce}-{i}.",
              "max_tokens": variants[i % 4][1]} for i, case in enumerate(data)]
    yield "composition", "mixed", mixed, 4, 128
    for concurrency in (4, 8, 16, 32):
        nonce = uuid.uuid4().hex
        items = [{**base[i % len(base)], "id": f"{base[i % len(base)]['id']}-{i}",
                  "system": f"{SYSTEM} Stress trial {nonce}-{i}."} for i in range(max(20, concurrency))]
        yield "stress", f"c-{concurrency}", items, concurrency, 64
    yield "recovery", "after-stress", base[:4], 1, 64
    nonce = uuid.uuid4().hex
    repeated = [{**prompt({**data[0], "id": f"shared-{i}", "question": x["question"]}),
                 "system": f"{SYSTEM} RAG trial {nonce}."} for i, x in enumerate(data[:8])]
    yield "rag", "same-document", repeated, 4, 64
    nonce = uuid.uuid4().hex
    yield "rag", "different-documents", [{**x, "system": f"{SYSTEM} RAG trial {nonce}."} for x in base[:8]], 4, 64
    nonce = uuid.uuid4().hex
    yield "rag", "long-context", [{**prompt(x, 4500), "system": f"{SYSTEM} RAG trial {nonce}."}
                                    for x in data[:8]], 4, 64


def main() -> None:
    import mlflow
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("artifacts/vllm_benchmark"))
    parser.add_argument("--only", choices=["baseline", "concurrency", "prefix", "context", "composition", "stress", "recovery", "rag"])
    parser.add_argument("--scenario", help="Run one named scenario within --only")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()
    model_info = httpx.get(args.base_url.rstrip("/") + "/v1/models", timeout=10).json()["data"][0]
    server_version = httpx.get(args.base_url.rstrip("/") + "/version", timeout=10).json().get("version", "unknown")
    if model_info["id"] != MODEL:
        raise RuntimeError(f"Unexpected vLLM model: {model_info['id']}")
    data = cases()
    args.output.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("techqa-vllm-inference")
    output = args.output / "measurements.jsonl"
    for experiment, name, items, concurrency, tokens in scenarios(data):
        if args.only and args.only != experiment:
            continue
        if args.scenario and args.scenario != name:
            continue
        print(f"running {experiment}/{name}", flush=True)
        with mlflow.start_run(run_name=f"{experiment}-{name}") as run:
            mlflow.log_params({"experiment": experiment, "scenario": name, "model": MODEL,
                               "vllm_version": server_version,
                               "max_model_len": model_info.get("max_model_len", "unknown"),
                               "concurrency": concurrency, "max_tokens": tokens,
                               "corpus_sha256": sha(CORPUS), "questions_sha256": sha(QUESTIONS),
                               "frozen_dense_run_id": "f284f735f18f489bb00cef2b4b417fed",
                               "cache_policy": "unique_nonce" if experiment in ("baseline", "concurrency", "prefix", "context", "composition", "stress", "rag") else "natural",
                               "ranking_sha256": sha(RANKING), "server_url": args.base_url})
            if experiment not in ("baseline", "prefix", "context") and not name.endswith("-warm"):
                stream_request(args.base_url, MODEL, items[0], min(tokens, 16), args.timeout)
            rows, details = run_batch(args.base_url, MODEL, items, concurrency, tokens, args.timeout)
            stats = summary(rows, details)
            record = {"run_id": run.info.run_id, "experiment": experiment, "scenario": name,
                      "concurrency": concurrency, "max_tokens": tokens, "summary": stats,
                      "details": details, "rows": rows}
            mlflow.log_metrics({k: v for k, v in stats.items() if isinstance(v, (int, float)) and v is not None})
            mlflow.log_dict(record, "measurements/record.json")
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps({"scenario": name, "run_id": run.info.run_id, **{k: v for k, v in stats.items() if k != "server_metric_delta"}}), flush=True)


if __name__ == "__main__":
    main()
