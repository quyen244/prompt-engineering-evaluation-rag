# TechQA evaluation

The lightweight TechQA inputs are `eval_data/corpus.jsonl` and `eval_data/eval_questions.json`.
The first file has 100 documents, and the second has 20 questions with reference answers and gold document IDs.
The original files remain in `D:\Downloads\TechQA\rag_eval_light\`.

## Retrieval ablation

Start Ollama with `qwen3-embedding:0.6b` installed.
The cached Hugging Face `intfloat/multilingual-e5-base` checkpoint is used offline.
The command below compares BM25, dense, and hybrid retrieval at Top-K 3, 5, and 10 across the two embedding models.
BM25 has no embedding model, so the matrix has 15 distinct configurations.

```powershell
docker compose run --rm --no-deps rag-mlflow python -m scripts.techqa_retrieval `
  --methods bm25 dense hybrid `
  --top-k 3 5 10 `
  --embeddings ollama:qwen3-embedding:0.6b huggingface:intfloat/multilingual-e5-base
```

The script validates the two files, splits documents into 850-character passages with 150-character overlap, and ranks documents by their best passage.
It logs Hit Rate, Precision, Recall, MRR, and nDCG plus per-question document and passage IDs to MLflow experiment `techqa-retrieval`.
It selects the highest observed nDCG, then Hit Rate, then MRR, then the smaller Top-K on ties.
The result is written to `eval_data/results/techqa_retrieval_champion.json` and tied to the dataset fingerprint and MLflow run ID.
On the supplied 20 questions, dense Qwen scored 1.000 for Hit Rate, Precision, Recall, MRR, and nDCG at Top-K 1 in MLflow experiment `techqa-retrieval-k1-confirmation`.
This K1 run followed the full 15-configuration matrix and supplies the champion for generation.

```powershell
docker compose run --rm --no-deps rag-mlflow python -m scripts.techqa_retrieval `
  --methods dense --top-k 1 --embeddings ollama:qwen3-embedding:0.6b `
  --experiment techqa-retrieval-k1-confirmation
```

## Generation evaluation

Run this after retrieval has produced the champion file.
The generator is local Ollama `qwen3.5:4b`.
The default independent judge is OpenRouter `inclusionai/ling-3.0-flash-fin:free`, using `OPENROUTER_API_KEY` from `.env`.
The other published IDs in the request are `stealth/space-bunny-alpha` and `nvidia/nemotron-3.5-lightning:free`.
Invoke either in a separate run with `--judge-model` so scores from different judges are not mixed.
The Ling judge request disables reasoning because its reasoning output otherwise consumed the response token budget before returning a JSON score.

```powershell
docker compose run --rm --no-deps rag-mlflow python -m scripts.techqa_generation
```

The evaluator uses the measured retrieval method, embedding and Top-K from the champion file.
It scores faithfulness, answer correctness, answer relevancy and citation precision with an explicit judge rubric.
These are rubric scores, not RAGAS scores.
Context precision and context recall use the supplied single gold document ID and are deterministic document-level metrics.
MLflow experiment `techqa-generation` stores model and dataset provenance, aggregate metrics, timing and per-case IDs/scores without raw question or answer text.

## Live query load

The load setup runs after generation evaluation and uses the current FastAPI serving path.
That path uses Qdrant dense retrieval with local Qwen embedding, so its load results are a serving baseline if another retrieval configuration wins offline.
The script creates a dedicated TechQA Qdrant collection, then runs 20 sequential queries with one user and 100 concurrent queries with 100 users.
It records response time percentiles, throughput, failures, and embedding/retrieval/generation stage timings in MLflow.

```powershell
./scripts/run_techqa_load.ps1
```

The 100-user step starts only after the one-user step completes successfully.
Each run has request and duration limits configured in the script.

## Interpretation

The 20 supplied questions are used to select the retrieval champion and to score generation, so these measurements do not establish generalization to unseen questions.
Use an independent holdout before claiming a production improvement.
