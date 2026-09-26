"""Lightweight contracts for TechQA generation judge parsing and evidence."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from scripts.techqa_generation import (
    RetryableJudgeError, _source_context, _sources, generate_answer, judge_answer,
    load_champion, parse_judge_scores,
)


class TechQAGenerationTests(unittest.TestCase):
    def test_parse_judge_scores_accepts_fenced_json(self) -> None:
        result = parse_judge_scores(
            '```json\n{"faithfulness": 1, "answer_correctness": 0.75, '
            '"answer_relevancy": 0.9, "citation_precision": 0}\n```'
        )
        self.assertEqual(result["faithfulness"], 1.0)
        self.assertEqual(result["citation_precision"], 0.0)

    def test_parse_judge_scores_rejects_missing_or_invalid_values(self) -> None:
        for content in (
            "not JSON",
            '{"faithfulness": 1}',
            '{"faithfulness": 1, "answer_correctness": 2, '
            '"answer_relevancy": 0.5, "citation_precision": 0.5}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_judge_scores(content)

    def test_sources_preserves_real_doc_ids(self) -> None:
        context = _sources(
            [{"doc_id": "swg123", "title": "IBM note", "text": "Known resolution"}], 500
        )
        self.assertIn("[doc:swg123]", context)
        self.assertIn("Known resolution", context)

    def test_query_aware_excerpt_keeps_middle_evidence(self) -> None:
        text = "unrelated filler " * 250 + "CVE-2026-4242 fix is APAR PH12345. " + "unrelated filler " * 250
        context, provenance = _source_context(
            [{"doc_id": "swg123", "title": "IBM note", "text": text}],
            "What fixes CVE-2026-4242?", 1200,
        )
        self.assertIn("APAR PH12345", context)
        self.assertLess(len(context), len(text))
        self.assertGreater(len(provenance[0]["spans"]), 1)

    def test_retrieved_chunk_is_first_and_extra_evidence_is_traced(self) -> None:
        text = "A" * 850 + "B" * 700 + "Resolution for CVE-2026-4242 is PH12345. " + "C" * 900
        doc = {
            "doc_id": "swg123", "title": "IBM note", "text": text,
            "chunk_id": "swg123::chunk-0000",
            "context_text": "IBM note\n\n" + text[:850],
        }
        context, provenance = _source_context([doc], "CVE-2026-4242 resolution", 2100)
        self.assertIn(doc["context_text"], context)
        self.assertLess(context.index(doc["context_text"]), context.index("PH12345"))
        self.assertEqual(provenance[0]["retrieved_chunk_id"], doc["chunk_id"])
        self.assertEqual(provenance[0]["retrieved_chunk_span"], [0, 850])
        self.assertTrue(provenance[0]["extra_spans"])

    def test_retrieved_chunk_must_match_corpus(self) -> None:
        doc = {"doc_id": "swg123", "title": "IBM note", "text": "real text",
               "chunk_id": "swg123::chunk-0000", "context_text": "IBM note\n\nwrong"}
        with self.assertRaisesRegex(ValueError, "does not match corpus"):
            _source_context([doc], "question", 1200)

    def test_generator_uses_native_ollama_without_thinking(self) -> None:
        response = {"message": {"content": "Grounded answer [doc:swg123]", "thinking": ""},
                    "prompt_eval_count": 42, "eval_count": 12}
        docs = [{"doc_id": "swg123", "title": "IBM note", "text": "answer source"}]
        with patch("scripts.techqa_generation._chat", return_value=response) as chat:
            answer, usage = generate_answer(
                "question", docs, model="qwen3.5:4b", ollama_base_url="http://localhost:11434",
                max_context_chars=500, max_tokens=256, timeout=5,
            )
        self.assertEqual(answer, "Grounded answer [doc:swg123]")
        self.assertEqual(usage, {"prompt_tokens": 42, "completion_tokens": 12})
        self.assertEqual(chat.call_args.args[1], "/api/chat")
        self.assertIs(chat.call_args.args[2]["think"], False)
        self.assertEqual(chat.call_args.args[2]["options"]["num_predict"], 256)

    def test_judge_retries_malformed_json_same_model(self) -> None:
        valid = {"choices": [{"message": {"content": (
            '{"faithfulness":1,"answer_correctness":1,'
            '"answer_relevancy":1,"citation_precision":1}'
        )}}], "usage": {}}
        bad = {"choices": [{"message": {"content": "bad"}}]}
        with patch("scripts.techqa_generation._chat", side_effect=[bad, valid]) as chat, \
             patch("scripts.techqa_generation.time.sleep"):
            scores, _ = judge_answer(
                "question", "reference", "answer", [], model="judge/a",
                api_key="test", openrouter_base_url="https://example.test",
                max_context_chars=500, timeout=1,
            )
        self.assertEqual(scores["faithfulness"], 1.0)
        self.assertEqual([call.args[2]["model"] for call in chat.call_args_list],
                         ["judge/a", "judge/a"])
        self.assertEqual(chat.call_args.args[2]["reasoning"], {"enabled": False})

    def test_judge_retries_rate_limit(self) -> None:
        valid = {"choices": [{"message": {"content": (
            '{"faithfulness":0,"answer_correctness":0,'
            '"answer_relevancy":0,"citation_precision":0}'
        )}}]}
        with patch("scripts.techqa_generation._chat",
                   side_effect=[RetryableJudgeError(429, 2), valid]) as chat, \
             patch("scripts.techqa_generation.time.sleep") as sleep:
            judge_answer("q", "r", "a", [], model="judge/a", api_key="test",
                         openrouter_base_url="https://example.test",
                         max_context_chars=500, timeout=1)
        self.assertEqual(chat.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_champion_must_match_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "champion.json"
            path.write_text(json.dumps({
                "method": "bm25", "embedding": "ollama:qwen3-embedding:0.6b",
                "top_k": 3, "dataset_fingerprint": "same", "retrieval_run_id": "run-1",
            }), encoding="utf-8")
            self.assertEqual(load_champion(path, "same")["top_k"], 3)
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                load_champion(path, "different")


if __name__ == "__main__":
    unittest.main()
