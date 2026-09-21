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
