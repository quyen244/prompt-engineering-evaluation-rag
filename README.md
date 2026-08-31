# RAG Production System — Prompt Engineering & Evaluation

Hệ thống RAG hoàn chỉnh: user upload PDF → hỏi đáp trên chính tài liệu đó.
Xây kèm evaluation harness, A/B testing có số liệu, model/prompt registry,
observability đầy đủ và xử lý bất đồng bộ.

> 🚧 Đang xây theo lộ trình 8 phase. Xem **[ROADMAP.md](ROADMAP.md)**.

---

## Bắt đầu

```powershell
# 1. Môi trường (Python 3.11)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. Torch CPU-only TRƯỚC (nếu không pip sẽ kéo bản CUDA ~2.5GB vô ích)
pip install --index-url https://download.pytorch.org/whl/cpu torch

# 3. Phần còn lại
pip install -r requirements/base.txt

# 4. Cấu hình
copy .env.example .env
#    mở .env và điền OPENROUTER_API_KEY

# 5. Kiểm tra môi trường
python practice/00_setup/00_env_check.py
```

---

## Cấu trúc repo

```
ROADMAP.md              ← đọc file này trước
docs/                   hướng dẫn chi tiết từng phase
  00_setup.md               Phase 0 — nền móng
  01_evaluation.md          Phase 1 — eval harness
  02_prompt_optimization.md Phase 2 — tối ưu prompt
  03_ab_testing.md          Phase 3 — A/B testing
  04_serving.md             Phase 4 — MLflow + FastAPI
  05_observability.md       Phase 5 — OTel/Prometheus/Grafana
  06_async_celery.md        Phase 6 — Celery
  07_production.md          Phase 7 — ghép & deploy
  CV_TALKING_POINTS.md      cách kể project khi phỏng vấn

practice/               code học từng feature, chạy độc lập
  common.py                 hạ tầng dùng chung — ĐỌC TRƯỚC
  00_setup/ ... 06_async/   practice theo phase

src/                    hệ thống thật (bạn tự viết theo hướng dẫn trong docs/)
infra/                  cấu hình Prometheus / Grafana / Tempo
requirements/           dependency có pin version
data/                   tài liệu nguồn
eval_data/              golden dataset cho evaluation
```

---

## Cách học

1. Đọc `docs/0X_*.md` của phase.
2. Chạy từng file trong `practice/0X_*/` theo thứ tự số — mỗi file dạy **một** khái niệm.
3. Làm phần `# ===== BÀI TẬP =====` ở cuối mỗi file.
4. Theo mục "Ghép vào hệ thống" trong guide để tự viết code trong `src/`.
5. Check **Definition of Done** trước khi sang phase tiếp.

```powershell
python practice/00_setup/00_env_check.py
python practice/00_setup/01_openrouter_hello.py
python practice/00_setup/02_local_embedding.py
python practice/00_setup/03_embedding_compare.py
```

---

## Chạy hệ thống hiện tại

```powershell
streamlit run src/app.py     # giao diện web
python -m src.app_cli        # giao diện dòng lệnh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Kết quả đo được

> Điền dần qua các phase. Bảng này là thứ đi vào CV.

| Chỉ số | Baseline | Champion | Cải thiện |
|---|---|---|---|
| Hit Rate @5 | — | — | — |
| NDCG @5 | — | — | — |
| Faithfulness | — | — | — |
| p95 latency | — | — | — |
| Cost / 1000 query | — | — | — |

---

## Stack

**RAG**: LlamaIndex · Qdrant · sentence-transformers (local embedding) · OpenRouter (LLM)
**Eval & Experiment**: MLflow · LLM-as-judge tự viết · bootstrap CI
**Serving**: FastAPI · MLflow Model Registry · SSE streaming
**Observability**: OpenTelemetry · Prometheus · Grafana · Tempo · structlog
**Async**: Celery · Redis
