"""Build a fixed 50-question tune / 20-question test TechQA retrieval study."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


VERSION = "1.0.0"
TRAIN_SOURCE_URL = "https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/resolve/main/train.json"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _selected_technotes(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    """Stream the official ID/TITLE/TEXT file without loading its 2.7 GB into RAM."""
    documents: dict[str, dict[str, str]] = {}
    doc_id = ""
    title = ""
    text_lines: list[bytes] = []
    section = ""

    def finish() -> None:
        if doc_id not in wanted:
            return
        body = b"".join(text_lines).decode("utf-8").strip()
        if not title or not body:
            raise ValueError(f"TechQA document is missing title or text: {doc_id}")
        documents[doc_id] = {"doc_id": doc_id, "title": title, "text": body}

    with path.open("rb") as source:
        for line in source:
            marker = line.strip()
            if marker == b"====== ID ======":
                finish()
                if len(documents) == len(wanted):
                    break
                doc_id, title, text_lines, section = "", "", [], "id"
            elif marker == b"====== TITLE ======":
                section = "title"
            elif marker == b"====== TEXT ======":
                section = "text"
            elif section == "id" and marker:
                doc_id = marker.decode("utf-8")
            elif section == "title" and marker:
                title = marker.decode("utf-8")
            elif section == "text" and doc_id in wanted:
                text_lines.append(line)
        else:
            finish()
    return documents


def prepare(
    full_corpus: Path, train_source: Path, light_corpus: Path,
    test_questions: Path, output_dir: Path, *, seed: int = 20261014,
) -> dict[str, object]:
    for path in (full_corpus, train_source, light_corpus, test_questions):
        if not path.is_file():
            raise ValueError(f"input file missing: {path}")
    source_rows = json.loads(train_source.read_text(encoding="utf-8"))
    test_rows = json.loads(test_questions.read_text(encoding="utf-8"))
    light_rows = [json.loads(line) for line in light_corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(source_rows, list) or not isinstance(test_rows, list) or len(test_rows) != 20:
        raise ValueError("expected a TechQA source list and exactly 20 test questions")
    test_ids = {row["id"] for row in test_rows}
    if len(test_ids) != 20:
        raise ValueError("test question IDs must be unique")
    candidates = []
    for row in source_rows:
        if not row.get("id", "").startswith("TRAIN_") or row["id"] in test_ids:
            continue
        if row.get("is_impossible") or not row.get("contexts"):
            continue
        context = row["contexts"]
        if len(context) != 1 or not context[0].get("filename", "").endswith(".txt"):
            continue
        candidates.append({
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["answer"],
            "gold_doc_id": context[0]["filename"][:-4],
        })
    if len({row["id"] for row in candidates}) != len(candidates) or len(candidates) < 50:
        raise ValueError("fewer than 50 unique answerable training questions are available")
    random.Random(seed).shuffle(candidates)
    tune_rows = sorted(candidates[:50], key=lambda row: row["id"])
    needed_ids = {row["doc_id"] for row in light_rows}
    needed_ids.update(row["gold_doc_id"] for row in tune_rows)
    needed_ids.update(row["gold_doc_id"] for row in test_rows)
    documents = _selected_technotes(full_corpus, needed_ids)
    missing = needed_ids - documents.keys()
    if missing:
        raise ValueError(f"TechQA full corpus lacks {len(missing)} required document IDs: {', '.join(sorted(missing)[:5])}")
    for row in tune_rows:
        if row["ground_truth"] not in documents[row["gold_doc_id"]]["text"]:
            raise ValueError(f"TechQA answer span is absent from gold document: {row['id']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_out = output_dir / "corpus.jsonl"
    tune_out = output_dir / "tune_questions.json"
    test_out = output_dir / "test_questions.json"
    corpus_out.write_text("".join(json.dumps(documents[doc_id], ensure_ascii=False) + "\n"
                                  for doc_id in sorted(documents)), encoding="utf-8")
    tune_out.write_text(json.dumps(tune_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    test_out.write_text(json.dumps(test_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "seed": seed,
        "train_source_url": TRAIN_SOURCE_URL,
        "train_source_sha256": _sha256(train_source),
        "full_corpus_path": str(full_corpus.resolve()),
        "full_corpus_bytes": full_corpus.stat().st_size,
        "light_corpus_sha256": _sha256(light_corpus),
        "document_count": len(documents),
        "tune_question_count": len(tune_rows),
        "test_question_count": len(test_rows),
        "tune_question_ids": [row["id"] for row in tune_rows],
        "test_question_ids": [row["id"] for row in test_rows],
        "corpus_sha256": _sha256(corpus_out),
        "tune_questions_sha256": _sha256(tune_out),
        "test_questions_sha256": _sha256(test_out),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv in (["-v"], ["-V"], ["--version"]):
        print(VERSION)
        return 0
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--full-corpus", type=Path, default=Path("D:/Downloads/TechQA/TechQA/technote_corpus/technotes.id_title_text"))
    parser.add_argument("--train-source", type=Path, default=Path("data/techqa_rag_eval_train.json"))
    parser.add_argument("--light-corpus", type=Path, default=Path("eval_data/corpus.jsonl"))
    parser.add_argument("--test-questions", type=Path, default=Path("eval_data/eval_questions.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("eval_data/rrf_study"))
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.full_corpus, args.train_source, args.light_corpus,
                         args.test_questions, args.output_dir, seed=args.seed)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        print("help: Check the four input paths and TechQA ID/TITLE/TEXT format")
        return 1
    print(f"study: {args.output_dir}")
    print(f"documents: {result['document_count']}")
    print(f"tune_questions: {result['tune_question_count']}")
    print(f"test_questions: {result['test_question_count']}")
    print(f"corpus_sha256: {result['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
