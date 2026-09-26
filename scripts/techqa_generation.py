"""Run TechQA answer evaluation with local Ollama and an OpenRouter judge.

The metrics here are explicitly LLM-judge rubric scores, not RAGAS scores.
Only aggregate and non-textual per-case evidence is persisted to MLflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from scripts.techqa_retrieval import CHUNK_OVERLAP, CHUNK_SIZE, build_retriever, load_dataset


PRIMARY_JUDGE = "inclusionai/ling-3.0-flash-fin:free"
ALTERNATE_JUDGES = ("stealth/space-bunny-alpha", "nvidia/nemotron-3.5-lightning:free")
SCORE_NAMES = ("faithfulness", "answer_correctness", "answer_relevancy", "citation_precision")
PROMPT_VERSION = "techqa-grounded-v1"
JUDGE_RUBRIC_VERSION = "techqa-judge-v1"
CONTEXT_POLICY_VERSION = "retrieved-chunk-first-question-windows-v2"


class RetryableJudgeError(RuntimeError):
    def __init__(self, status: int, retry_after: float | None = None) -> None:
        super().__init__(f"OpenRouter HTTP {status}; retry after a short delay")
        self.retry_after = retry_after


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chat(base_url: str, path: str, payload: dict[str, Any], *, api_key: str | None = None,
          timeout: float = 120) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{base_url.rstrip('/')}{path}", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if api_key and (status == 429 or status >= 500):
            retry_header = exc.response.headers.get("retry-after", "")
            try:
                retry_after = float(retry_header)
            except ValueError:
                retry_after = None
            raise RetryableJudgeError(status, retry_after) from exc
        raise RuntimeError(f"HTTP {status} from {base_url}; verify model ID, quota and credentials") from exc
    except (httpx.RequestError, ValueError) as exc:
        raise RuntimeError(f"Cannot get a valid response from {base_url}; check service and network") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid response shape from {base_url}")
    return data


def _content(data: dict[str, Any]) -> str:
    try:
        value = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Model response has no message content") from exc
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("Model returned empty answer; increase max tokens or choose another model")
    return value.strip()


def _terms(question: str) -> set[str]:
    stop = {"about", "after", "allow", "does", "find", "from", "have", "help", "information",
            "into", "need", "that", "the", "their", "there", "this", "through", "what",
            "when", "where", "which", "with", "would", "your"}
    return {term for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) >= 3 and term not in stop}


def _excerpts(text: str, question: str, budget: int) -> list[tuple[int, int]]:
    """Keep head/tail plus high-overlap windows using only the question as a signal."""
    if len(text) <= budget:
        return [(0, len(text))]
    edge = min(400, max(80, budget // 6))
    chosen = [(0, edge), (len(text) - edge, len(text))]
    remaining = max(0, budget - 2 * edge)
    terms = _terms(question)
    for _ in range(2):
        width = remaining // 2
        if width < 80:
            break
        step = max(80, width // 3)
        starts = range(0, max(1, len(text) - width + 1), step)
        candidates = [*starts, max(0, len(text) - width)]
        def score(start: int) -> tuple[float, int]:
            end = min(len(text), start + width)
            if any(start < prior_end and end > prior_start for prior_start, prior_end in chosen):
                return (-1.0, -start)
            snippet = text[start:end].lower()
            overlap = sum(min(snippet.count(term), 3) for term in terms)
            headings = sum(snippet.count(term) for term in
                           ("problem conclusion", "local fix", "resolution", "solution", "answer"))
            return (overlap + 2 * headings, -start)
        best = max(candidates, key=score)
        if score(best)[0] < 0:
            break
        chosen.append((best, min(len(text), best + width)))
    return sorted(chosen)


def _retrieved_chunk_span(doc: dict[str, Any]) -> tuple[int, int]:
    match = re.fullmatch(rf"{re.escape(doc['doc_id'])}::chunk-(\d+)", doc["chunk_id"])
    if not match:
        raise ValueError(f"Invalid retrieved chunk ID for {doc['doc_id']}")
    start = int(match.group(1)) * (CHUNK_SIZE - CHUNK_OVERLAP)
    end = min(len(doc["text"]), start + CHUNK_SIZE)
    expected = f"{doc['title']}\n\n{doc['text'][start:end]}"
    if doc["context_text"] != expected:
        raise ValueError(f"Retrieved chunk text does not match corpus for {doc['doc_id']}")
    return start, end


def _extra_excerpts(text: str, question: str, budget: int,
                    excluded: tuple[int, int]) -> list[tuple[int, int]]:
    if budget < 160:
        return []
    width = max(80, min(650, budget // 2))
    step = max(80, width // 3)
    candidates = list(range(0, max(1, len(text) - width + 1), step))
    candidates.append(max(0, len(text) - width))
    chosen = [excluded]
    terms = _terms(question)
    extras: list[tuple[int, int]] = []
    for _ in range(2):
        eligible = []
        for start in candidates:
            end = min(len(text), start + width)
            if any(start < previous_end and end > previous_start
                   for previous_start, previous_end in chosen):
                continue
            snippet = text[start:end].lower()
            overlap = sum(min(snippet.count(term), 3) for term in terms)
            headings = sum(snippet.count(term) for term in
                           ("problem conclusion", "local fix", "resolution", "solution", "answer"))
            eligible.append((overlap + 2 * headings, -start, start, end))
        if not eligible:
            break
        _, _, start, end = max(eligible)
        extras.append((start, end))
        chosen.append((start, end))
    return sorted(extras)


def _source_context(docs: list[dict[str, Any]], question: str,
                    max_context_chars: int) -> tuple[str, list[dict[str, Any]]]:
    if not docs:
        return "(No retrieved sources)", []
    per_doc = max_context_chars // len(docs)
    sections: list[str] = []
    provenance: list[dict[str, Any]] = []
    for doc in docs:
        header = f"[doc:{doc['doc_id']}]\n"
        if "context_text" in doc and "chunk_id" in doc:
            chunk_span = _retrieved_chunk_span(doc)
            primary = f"[retrieved chunk:{doc['chunk_id']}]\n{doc['context_text']}"
            extra_budget = per_doc - len(header) - len(primary) - 100
            if extra_budget < 0:
                raise ValueError("--max-context-chars cannot fit the retrieved chunks")
            extras = _extra_excerpts(doc["text"], question, extra_budget, chunk_span)
            extra_text = "".join(
                f"\n[extra chars {start}:{end}] {doc['text'][start:end]}"
                for start, end in extras
            )
            sections.append(header + primary + extra_text)
            provenance.append({
                "doc_id": doc["doc_id"], "text_chars": len(doc["text"]),
                "retrieved_chunk_id": doc["chunk_id"],
                "retrieved_chunk_span": list(chunk_span),
                "extra_spans": [[start, end] for start, end in extras],
            })
        else:
            text_budget = per_doc - len(header) - len(doc["title"]) - 100
            if text_budget < 240:
                raise ValueError("--max-context-chars too small for retrieved document count")
            spans = _excerpts(doc["text"], question, text_budget)
            excerpt = "\n[... omitted ...]\n".join(
                f"[chars {start}:{end}] {doc['text'][start:end]}" for start, end in spans
            )
            sections.append(header + doc["title"] + "\n" + excerpt)
            provenance.append({"doc_id": doc["doc_id"], "text_chars": len(doc["text"]),
                               "spans": [[start, end] for start, end in spans]})
    return "\n\n".join(sections), provenance


def _sources(docs: list[dict[str, Any]], max_context_chars: int,
             question: str = "") -> str:
    return _source_context(docs, question, max_context_chars)[0]


def generate_answer(question: str, docs: list[dict[str, Any]], *, model: str,
                    ollama_base_url: str, max_context_chars: int, max_tokens: int,
                    timeout: float) -> tuple[str, dict[str, Any]]:
    context = _sources(docs, max_context_chars, question)
    data = _chat(ollama_base_url, "/api/chat", {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Answer only from the supplied documents. Cite supporting facts as [doc:ID] "
                "using the exact document ID. If evidence is insufficient, say so. Be concise."
            )},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion:\n{question}"},
        ],
        "options": {"temperature": 0, "num_predict": max_tokens},
        "think": False,
        "stream": False,
    }, timeout=timeout)
    message = data.get("message", {})
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Ollama returned empty message.content; check model and think=false support")
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }
    return answer.strip(), usage


def parse_judge_scores(content: str) -> dict[str, float]:
    """Reject malformed or unbounded judge output instead of inventing scores."""
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError("judge did not return a JSON object")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("judge returned malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("judge returned a non-object")
    result: dict[str, float] = {}
    for name in SCORE_NAMES:
        value = parsed.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"judge score {name} must be a number in [0, 1]")
        result[name] = float(value)
    return result


def judge_answer(question: str, ground_truth: str, answer: str,
                 docs: list[dict[str, Any]], *, model: str, api_key: str,
                 openrouter_base_url: str, max_context_chars: int,
                 timeout: float) -> tuple[dict[str, float], dict[str, Any]]:
    context = _sources(docs, max_context_chars, question)
    rubric = (
        "You are an independent RAG evaluator. Return ONLY one JSON object with four numeric "
        "fields from 0 to 1: faithfulness, answer_correctness, answer_relevancy, citation_precision. "
        "Faithfulness: answer claims supported by retrieved documents. "
        "Answer correctness: factual agreement and completeness versus reference answer. "
        "Answer relevancy: directly answers the question. "
        "Citation precision: cited document IDs exist in supplied documents and support the "
        "attached claims; score 0 if no citations are present. "
        "Do not assume the reference answer is present in retrieved documents. "
        "Use 0 for fully poor and 1 for fully good. No prose or markdown."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": rubric},
            {"role": "user", "content": (
                f"Question:\n{question}\n\nReference answer:\n{ground_truth}\n\n"
                f"Retrieved documents:\n{context}\n\nGenerated answer:\n{answer}"
            )},
        ],
        "temperature": 0,
        "max_tokens": 512,
        "reasoning": {"enabled": False},
    }
    for attempt in range(3):
        try:
            data = _chat(openrouter_base_url, "/chat/completions", payload,
                         api_key=api_key, timeout=timeout)
            try:
                return parse_judge_scores(_content(data)), data.get("usage", {})
            except RuntimeError as exc:
                raise ValueError("judge returned empty or missing content") from exc
        except (RetryableJudgeError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"OpenRouter judge failed after 3 attempts: {exc}") from exc
            delay = min(8.0, max(1.0, getattr(exc, "retry_after", 0) or 2 ** attempt))
            time.sleep(delay)
    raise AssertionError("unreachable")


def _token_count(usage: dict[str, Any], field: str) -> int:
    value = usage.get(field, 0) if isinstance(usage, dict) else 0
    return value if isinstance(value, int) and value >= 0 else 0


def ollama_model_digest(base_url: str, model: str) -> str:
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            models = response.json()["models"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Cannot query Ollama models; start Ollama and check --ollama-url") from exc
    for entry in models:
        if entry.get("name") == model:
            return str(entry.get("digest", "unknown"))
    raise RuntimeError(f"Ollama generator model {model} is not installed; run ollama pull {model}")


def load_champion(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"Retrieval champion missing at {path}; run TechQA retrieval ablation first"
        )
    try:
        champion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read retrieval champion at {path}") from exc
    if not isinstance(champion, dict) or champion.get("dataset_fingerprint") != fingerprint:
        raise RuntimeError("Retrieval champion dataset fingerprint differs from current TechQA inputs")
    required = ("method", "embedding", "top_k", "retrieval_run_id")
    if any(not champion.get(key) for key in required):
        raise RuntimeError(f"Retrieval champion at {path} is missing required fields")
    if champion["method"] not in ("dense", "bm25", "hybrid") or not isinstance(champion["top_k"], int) or champion["top_k"] < 1:
        raise RuntimeError(f"Retrieval champion at {path} has invalid retrieval configuration")
    return champion


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_context_chars < 300 or args.timeout <= 0 or args.generator_max_tokens < 1 or args.limit < 0:
        raise ValueError("--timeout and --generator-max-tokens must be positive; --limit must be >= 0; --max-context-chars must be >= 300")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing; set it in the environment or .env")
    corpus_path, questions_path = Path(args.corpus), Path(args.questions)
    documents, questions, fingerprint = load_dataset(corpus_path, questions_path)
    champion = load_champion(Path(args.champion), fingerprint)
    retrieval_method = args.retriever or champion["method"]
    embedding = args.embedding or champion["embedding"]
    top_k = args.k or champion["top_k"]
    if top_k < 1:
        raise ValueError("--k must be positive")
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError("No TechQA questions selected")
    generator_digest = ollama_model_digest(args.ollama_url, args.generator_model)

    import mlflow

    tracking_uri = args.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)
    retriever = build_retriever(
        documents, method=retrieval_method, embedding=embedding,
        ollama_base_url=args.ollama_url,
    )
    rows: list[dict[str, Any]] = []
    with mlflow.start_run(run_name="techqa-generation") as active_run:
        mlflow.log_params({
            "dataset_fingerprint": fingerprint,
            "corpus_sha256": _sha256(corpus_path),
            "questions_sha256": _sha256(questions_path),
            "corpus_count": len(documents),
            "question_count": len(questions),
            "generator_model": args.generator_model,
            "generator_digest": generator_digest,
            "retriever": retrieval_method,
            "embedding": embedding,
            "top_k": top_k,
            "retrieval_champion_run_id": champion["retrieval_run_id"],
            "retrieval_embedding_digest": champion.get("embedding_digest", "unknown"),
            "requested_judge_model": args.judge_model,
            "prompt_version": PROMPT_VERSION,
            "judge_rubric_version": JUDGE_RUBRIC_VERSION,
            "context_policy_version": CONTEXT_POLICY_VERSION,
            "max_context_chars": args.max_context_chars,
            "generator_max_tokens": args.generator_max_tokens,
        })
        mlflow.set_tag("evaluation_type", "llm_judge_rubric")
        mlflow.set_tag("context_metric_definition", "single_gold_document_id_precision_recall_at_k")
        try:
            for index, case in enumerate(questions, 1):
                start = time.perf_counter()
                docs = retriever.retrieve(case["question"], top_k)
                _, context_provenance = _source_context(
                    docs, case["question"], args.max_context_chars
                )
                retrieval_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                answer, generator_usage = generate_answer(
                    case["question"], docs, model=args.generator_model,
                    ollama_base_url=args.ollama_url,
                    max_context_chars=args.max_context_chars,
                    max_tokens=args.generator_max_tokens,
                    timeout=args.timeout,
                )
                generation_ms = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                scores, judge_usage = judge_answer(
                    case["question"], case["ground_truth"], answer, docs,
                    model=args.judge_model, api_key=api_key,
                    openrouter_base_url=args.openrouter_url,
                    max_context_chars=args.max_context_chars,
                    timeout=args.timeout,
                )
                judge_ms = (time.perf_counter() - start) * 1000
                cited_ids = re.findall(r"\[doc:([^\]]+)\]", answer)
                row = {
                    "case_id": case["id"],
                    "context_recall": int(case["gold_doc_id"] in {d["doc_id"] for d in docs}),
                    "context_precision": (
                        sum(d["doc_id"] == case["gold_doc_id"] for d in docs) / len(docs)
                        if docs else 0.0
                    ),
                    "retrieved_doc_ids": [d["doc_id"] for d in docs],
                    "context_spans": context_provenance,
                    "cited_doc_ids": cited_ids,
                    "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                    "answer_chars": len(answer),
                    "judge_model": args.judge_model,
                    "retrieval_ms": round(retrieval_ms, 3),
                    "generation_ms": round(generation_ms, 3),
                    "judge_ms": round(judge_ms, 3),
                    "generator_prompt_tokens": _token_count(generator_usage, "prompt_tokens"),
                    "generator_completion_tokens": _token_count(generator_usage, "completion_tokens"),
                    "judge_prompt_tokens": _token_count(judge_usage, "prompt_tokens"),
                    "judge_completion_tokens": _token_count(judge_usage, "completion_tokens"),
                    **scores,
                }
                rows.append(row)
                print(f"progress: {index}/{len(questions)}", file=sys.stderr)
            mlflow.set_tag("effective_judge_model", args.judge_model)
            metrics = {
                name: sum(row[name] for row in rows) / len(rows)
                for name in (*SCORE_NAMES, "context_precision", "context_recall")
            }
            metrics.update({
                "retrieval_ms_mean": sum(row["retrieval_ms"] for row in rows) / len(rows),
                "generation_ms_mean": sum(row["generation_ms"] for row in rows) / len(rows),
                "judge_ms_mean": sum(row["judge_ms"] for row in rows) / len(rows),
            })
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmp:
                artifact = Path(tmp) / "per_case.jsonl"
                artifact.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
                mlflow.log_artifact(str(artifact), artifact_path="generation")
            return {"run_id": active_run.info.run_id, "cases": len(rows),
                    "judge_model": args.judge_model, **metrics}
        except Exception:
            mlflow.set_tag("evaluation_status", "failed")
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLflow-tracked TechQA generation evaluation")
    parser.add_argument("--corpus", default="eval_data/corpus.jsonl")
    parser.add_argument("--questions", default="eval_data/eval_questions.json")
    parser.add_argument("--champion", default="eval_data/results/techqa_retrieval_champion.json")
    parser.add_argument("--retriever", choices=("dense", "bm25", "hybrid"), default=None,
                        help="Override measured champion method")
    parser.add_argument("--embedding", default=None, help="Override measured champion embedding")
    parser.add_argument("--k", type=int, default=None, help="Override measured champion top-k")
    parser.add_argument("--generator-model", default="qwen3.5:4b")
    parser.add_argument("--judge-model", default=PRIMARY_JUDGE,
                        help=f"OpenRouter judge; alternate verified IDs: {', '.join(ALTERNATE_JUDGES)}")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--openrouter-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""))
    parser.add_argument("--experiment", default="techqa-generation")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--generator-max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--limit", type=int, default=0, help="Run first N cases for a small smoke test")
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    try:
        result = run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stdout)
        print("help: Check model services, OPENROUTER_API_KEY and CLI flags, then rerun.", file=sys.stdout)
        return 1
    print(f"run_id: {result['run_id']}")
    print(f"cases: {result['cases']}")
    print(f"judge_model: {result['judge_model']}")
    for name in SCORE_NAMES:
        print(f"{name}: {result[name]:.4f}")
    print(f"context_precision: {result['context_precision']:.4f}")
    print(f"context_recall: {result['context_recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
