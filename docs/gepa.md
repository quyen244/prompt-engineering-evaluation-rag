# GepaPromptOptimizer - MLflow GenAI

## Định nghĩa
Thuật toán tối ưu hóa prompt không sử dụng gradient, thuộc họ GEPA (Gradient-Free Evolutionary Prompt Optimization). MLflow đóng gói GEPA và DSPy dưới cùng API `mlflow.genai.optimize_prompts()` [citation:8].

## Cách hoạt động
1. Đánh giá prompt hiện tại trên train_data
2. Dùng `reflection_model` (LLM mạnh) để phân tích lỗi
3. Sinh prompt mới khắc phục lỗi
4. Chọn prompt tốt nhất qua scorers
5. Lặp lại đến `max_metric_calls` [citation:3][citation:8]

## API
```python
from mlflow.genai.optimize import GepaPromptOptimizer

optimizer = GepaPromptOptimizer(
    reflection_model="openai:/gpt-4o",  # Bắt buộc
    max_metric_calls=100,               # Tối đa số lần đánh giá
    display_progress_bar=True
)

result = mlflow.genai.optimize_prompts(
    predict_fn=qa_pred_fn,
    train_data=train_dataset,
    prompt_uris=[prompt.uri],
    optimizer=optimizer,
    scorers=[custom_scorer],            # Tùy chọn
    enable_tracking=True
)
```

## RAG lineage adapter

The production-facing adapter is `src/rag/optimization/gepa_adapter.py`. It
accepts a LangChain `Runnable` and frozen `RetrievalSnapshot` values, so GEPA
can vary only the prompt while every candidate sees identical retrieved
context. `freeze_retrieval()` must run before optimization; candidate outputs
are retained in the append-only `OptimizationLedger` and can be logged as
per-case artifacts.

`src/rag/lineage.py` builds a content-addressed immutable manifest containing
retrieval strategy (BM25, reranker, hybrid), prompt version/digest, model,
dataset, budget, rollback, and run metadata. Prompt aliases are separate movable records.
Promotion requires a real non-empty held-out dataset, frozen retrieval,
recorded candidates, per-case artifacts, and an observed candidate score above
the baseline. Missing evidence raises `PromotionRefused`; no score is
fabricated.

The adapter and manifest modules are importable offline. Running tracked
optimization additionally requires the dependencies in `requirements/base.txt`.

## Frozen-RAG experiment contract

The optimization boundary is deliberately narrow:

1. `freeze_retrieval(cases, retrieve)` materializes one immutable
   `RetrievalSnapshot` per case before GEPA starts.
2. `GepaLangChainAdapter` accepts only those snapshots and a runnable with
   `invoke()`. It has no retriever callback, so candidate evaluation cannot
   silently refresh context.
3. `frozen_retrieval_contract` records the snapshot count, case IDs, and a
   content digest. The digest is logged with the candidate ledger and selection
   record.
4. `RunContract` binds train and held-out dataset digests, prompt/model
   identities, evaluation protocol, retrieval snapshot identity, and optional
   reflection model into one reproducible content-addressed artifact.

Candidate outputs are append-only `CandidateResult` records. Once an observed
held-out evaluation exists, `record_selection()` emits a signed-by-content
`GepaSelectionRecord` containing the baseline and selected scores, candidate
count, and run identity. Its digest is content-addressed and it does not claim
an improvement when no score was provided.

Selection and promotion are separate boundaries. `require_promotion()` checks
real held-out evidence, frozen retrieval, candidate records, and per-case
artifacts. `require_approved_promotion()` additionally requires an explicit
`PromotionApproval(approved_by, reason, approval_reference)` before an external
caller moves a mutable MLflow prompt alias. Rollback metadata identifies the
previous immutable prompt version, alias, run, and reason; no automatic alias
mutation or retriever change is performed here.

## Validation evidence

The focused tests exercise behavior through the public contracts: immutable
manifest and run-contract digests, train/held-out separation, frozen retrieval
call counts, cache behavior, candidate selection records, rejection of empty or
non-numeric held-out evidence, and the approval boundary. Run them from the
repository root after installing `requirements/base.txt`:

```sh
python -m pytest -q tests/test_lineage.py tests/test_gepa_adapter.py
python -m compileall -q src tests
```

The tests intentionally do not report quality numbers or fabricate GEPA metrics;
scores are accepted only from the caller's real scorer.
