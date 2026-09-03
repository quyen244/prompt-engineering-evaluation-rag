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