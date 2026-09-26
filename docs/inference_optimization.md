# Inference optimization study

This document records what was measured, which optimizations were applied, and which candidate techniques were rejected after measurement.
Every number here comes from an MLflow run in experiment `rag-query-concurrency` or from the Ollama server log, not from estimation.

Hardware for all runs: single NVIDIA GTX 1660 Ti with 6 GB VRAM, local Ollama serving `qwen3.5:4b` for generation and `qwen3-embedding:0.6b` for embedding, Qdrant in Docker, FastAPI in Docker.
Dataset: the 100-document TechQA corpus and its 20 questions, collection `techqa_eval_e2312c3ce80c736e`.

## Measured results

All four runs completed 20/20 requests with zero failures and terminated on `stop_reason=request_cap`.

| Run | Users | Top-K | Response mean | Response p50 | Generation mean | Embedding mean | Retrieval mean |
|---|---|---|---|---|---|---|---|
| `techqa-1-user` | 1 | 5 | 12 536 ms | 11 000 ms | 10 311 ms | 1 224 ms | 58 ms |
| `techqa-1-user-topk1` | 1 | 1 | **4 054 ms** | **4 000 ms** | **3 074 ms** | 623 ms | 48 ms |
| `techqa-20-users` | 20 | 5 | 88 568 ms | 94 000 ms | 70 989 ms | 7 234 ms | 1 802 ms |
| `techqa-20-users-topk1` | 20 | 1 | 112 214 ms | 117 000 ms | 60 283 ms | 40 905 ms | 2 610 ms |

Ingestion of the corpus into Qdrant (463 points) took 160.7 s, roughly 347 ms per point, dominated by embedding calls to Ollama.

### Run-to-run variance

Two runs of the identical configuration (20 users, Top-K 5) differed by 16% in total response time and 22% in generation time.
Any single-run improvement smaller than roughly 20% at 20 users cannot be distinguished from noise.
This variance figure is the reason the Top-K conclusion below is drawn from the 1-user runs, where queueing does not contribute.

## Applied optimizations

### Top-K reduced from 5 to 1

**Result: 3.1x faster end to end at 1 user** (12 536 ms to 4 054 ms, -67.7%), with generation alone 3.4x faster (-70.2%).
The effect far exceeds the ±20% noise band, so it is a real effect rather than run-to-run luck.

**Why this is safe.** The retrieval ablation established Top-K 1 as champion with Hit Rate, Precision, Recall, MRR and nDCG all 1.000 across 20/20 questions - the gold document ranks first every time.
The generation evaluation (`techqa-generation`, run `89bb770c`) was itself already executed at `top_k=1` and scored faithfulness 0.9525 and answer correctness 0.9425.
Serving at Top-K 5 therefore paid five times the context for no measured retrieval or answer-quality benefit.

**Why it works.** Generation time is prefill plus decode, and prefill scales with input token count.
Cutting retrieved context to one document cuts most of the prompt, so it cuts prefill directly.

**Boundary.** The gain does not survive at 20 concurrent users, where GPU contention dominates - see the rejected and open items below.

### MLflow server memory

**Result: 3.66 GiB down to 361 MiB, a 90% reduction**, verified with `docker stats` and by confirming all eight experiments still read back over the REST API.

MLflow 3.16 `mlflow ui` spawns thirteen processes: four Uvicorn workers, a job runner, and seven Huey job-queue workers.
Each inherits this repository's full ML image (torch, transformers, langchain), so each costs roughly 400 MB.
Tracking - logging params, metrics and artifacts - does not use the job subsystem at all.
Setting `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` and `--workers 1` in `docker-compose.yml` removes twelve of those processes.

### Prometheus histogram buckets

The `rag_query_duration_seconds` histogram had 40 s as its largest finite bucket.
`histogram_quantile` cannot extrapolate past that bucket, so it returned exactly 40 for every quantile above it, and the Grafana panel showed a flat line at 40 s while real generation p95 was 163 s.
Buckets now extend to 60, 120, 240 and 480 s.

A dashboard that plateaus at exactly a round number is usually saturating, not stable.

## Rejected after measurement

### KV cache quantization - rejected

The intuition was that concurrent requests inflate the KV cache and exhaust VRAM.
The Ollama server log disproves it:

```
llama_kv_cache: size = 128.00 MiB (4096 cells, 8 layers, 1/1 seqs), K (f16): 64.00 MiB, V (f16): 64.00 MiB
```

The KV cache is 128 MiB, which is 2% of the 6 GB card.
Quantizing to `q8_0` would recover about 64 MiB.
`OLLAMA_KV_CACHE_TYPE` is a real lever on machines where context length and parallel slots are large, but here the context is already capped at 4096 tokens with a single slot, so there is nothing to recover.

No benchmark was run for this change, because the memory accounting already shows the ceiling on any possible gain.

### Adding Redis or a message queue for queries - rejected as a speed measure

Redis and Celery already run in this stack and already serve PDF ingestion (`src/server.py`, `celery_app.send_task`).
The query path is synchronous by design.

Ollama already queues: `OLLAMA_MAX_QUEUE` is 512 and `OLLAMA_NUM_PARALLEL` is 1, so requests are already strictly serialized.
Putting another queue in front of a queue does not increase throughput, because the bottleneck is GPU compute, not the absence of a waiting line.

A queue remains worthwhile for **user experience**: it turns an opaque 112-second hang into a visible position in line, and it allows shedding load instead of accepting work that cannot be served.
It should be proposed as a UX and stability change, never as a latency optimization.

## VRAM accounting

From the Ollama log, for `qwen3.5:4b` on this card:

```
CUDA0 (GTX 1660 Ti) | 6143 total = 5076 free + (2759 = 2513 model + 178 context + 68 compute)
```

Model weights dominate at 2513 MiB.
Context, which includes the KV cache, is 178 MiB.
This is the accounting that rejects the KV cache hypothesis and that makes the parallel-slot option below affordable.

### `OLLAMA_NUM_PARALLEL` - blocked by model architecture, not by memory

Setting `OLLAMA_NUM_PARALLEL=4` and restarting Ollama does reach the server config, but the models still load with a single slot:

```
WARN msg="model architecture does not currently support parallel requests" architecture=qwen35
llama_context: n_seq_max = 1
srv load_model: initializing, n_slots = 1, n_ctx_slot = 4096
```

Memory is explicitly not the constraint - the same load reports `will leave 2316 >= 1685 MiB of free device memory, no changes needed`.

`qwen3.5:4b` is a hybrid architecture: the log shows 8 attention layers alongside 32 `llama_memory_recurrent` layers.
Recurrent layers carry per-sequence state that llama.cpp cannot yet share across parallel slots, so Ollama refuses parallel requests for this architecture and silently clamps the setting to 1.

This means the serving stack processes exactly one generation at a time regardless of configuration, which fully explains the 20-user latencies without needing any KV-cache or thrashing hypothesis.

The run was executed anyway to confirm the log with measurement (`techqa-20-users-parallel4`, 20/20 requests, zero failures, `complete=true`):

| Metric | parallel=1 | parallel=4 |
|---|---|---|
| Response mean | 88 568 ms | 105 619 ms |
| Response p50 | 94 000 ms | 108 000 ms |
| Throughput | 0.2 req/s | 0.1 req/s |
| Slots allocated | 1 | 1 |

No improvement was measured.
The differences sit inside the ±20% run-to-run noise band, so the honest reading is "no measurable effect", not "made it worse".

**Consequence for concurrency.** Real parallelism on this stack requires switching to a conventional transformer architecture that llama.cpp can batch - for example a Qwen2.5 or Llama-family model small enough for 6 GB.
That is a model change with answer-quality implications, so it must be re-scored with the judge against the same 20 questions, not swapped in on performance grounds alone.

## Open candidates, not yet measured

### Move query embedding off the GPU

Embedding one short question is roughly 415 ms at the median with no contention but 6 858 ms at 20 users, a 10x inflation, while Qdrant retrieval does not inflate at all because it does not share the GPU.
Running the 0.6B embedding model on CPU in-process would remove one of the two GPU contenders entirely.

The vectors must match those already indexed - same model, same normalization - so this requires re-verifying retrieval against the 20 questions before it can be trusted.

### Reduce `max_tokens`

Generation is currently capped at 512 output tokens and decode time is close to linear in tokens produced.
This trades answer completeness for speed, so it must be re-scored with the judge rather than assumed neutral.

## Method notes

Measure before optimizing, and measure the specific effect where noise is lowest.
The Top-K effect was invisible at 20 users, where an unrelated 465% embedding regression swamped it, and unambiguous at 1 user.
Choosing the quieter operating point is what made the conclusion possible.

Three measurement defects had to be fixed before any of these numbers could be trusted, and each was silent - no crash, no failing test:

- A local `response` variable in `src/server.py` shadowed the FastAPI `Response` parameter, so the `Server-Timing` header never reached the client and per-stage metrics were never collected, despite the feature being ticked complete.
- `runner.quit()` called from inside a Locust task greenlet deadlocked, so `test_stop` never completed and MLflow metrics were lost entirely.
- Locust runs under gevent, whose greenlets do not share MLflow's thread-local active-run stack, so params and metrics landed in two different runs and neither was reproducible on its own. All MLflow writes now go through an explicit `MlflowClient` and run id.
