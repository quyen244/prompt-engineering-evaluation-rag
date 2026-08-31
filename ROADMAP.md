# ROADMAP — Từ RAG baseline → Hệ thống Production sẵn sàng cho user

> **North Star**: Một hệ thống chạy thật, user upload file PDF → hỏi đáp trên chính file đó.
> Xung quanh nó là: evaluation tự động, A/B testing có số liệu, model/prompt registry,
> observability đầy đủ, và xử lý ngầm bất đồng bộ.
>
> **Mục tiêu phụ (quan trọng không kém)**: mỗi thành phần phải kể được thành một câu
> trong CV, kèm **con số**. Không có số = không tính.

---

## 0. Điểm xuất phát (đã có)

| Thành phần | Trạng thái |
|---|---|
| RAG baseline (LlamaIndex + OpenRouter) | ✅ chạy được |
| `ChunkerFactory` (4 loại chunker) | ✅ đã có — nền tốt cho A/B |
| Streamlit UI + CLI | ✅ chạy được |
| MLflow | ⚠️ mới thử với sklearn, chưa gắn vào RAG |
| Eval dataset | ⚠️ 5 câu hardcode, chưa có harness |
| Embedding | ❌ **đang sai** — xem "Nợ kỹ thuật" bên dưới |
| Vector store | ❌ in-memory, mất khi restart |
| API | ❌ chưa có |
| Observability | ❌ chưa có |
| Async | ❌ chưa có |

### Nợ kỹ thuật phải trả trước (Phase 0)

1. **Embedding đang gọi sai endpoint.** `src/rag/rag_chatbot.py` và `src/rag/main.py`
   dùng `OpenAIEmbedding(api_base="https://openrouter.ai/api/v1")`. OpenRouter chỉ
   proxy `/chat/completions` — **không có `/embeddings`**. Đây là lỗi im lặng nguy hiểm
   nhất trong project: hoặc nó throw, hoặc nó fallback về OpenAI thật và tiêu tiền
   của một API key không tồn tại. → Chuyển sang **HuggingFace embedding local**.
2. **`VectorStoreIndex` in-memory** — rebuild toàn bộ index mỗi lần khởi động. Với PDF
   thật (200 trang) là không chấp nhận được. → **Qdrant** persistent.
3. **Singleton `_chatbot_instance`** bỏ qua tham số khi đã khởi tạo — A/B testing sẽ
   âm thầm dùng lại config cũ và cho ra kết quả giả. → Bỏ singleton, dùng factory.
4. **Hardcode đường dẫn tuyệt đối** trong `src/rag/main.py` — không chạy được trên máy khác.
5. **Python version**: thư mục `venv_311` thực chất chứa **Python 3.14**, không phải 3.11.
   Sẽ tạo `.venv` mới bằng `py -3.11`.

---

## 1. Kiến trúc đích

```
                          ┌──────────────────────────────┐
                          │   Streamlit UI  /  curl      │
                          └──────────────┬───────────────┘
                                         │ HTTP
                          ┌──────────────▼───────────────┐
                          │        FastAPI  :8000        │
                          │  POST /ingest   (→ job_id)   │
                          │  POST /query    (streaming)  │
                          │  GET  /health   /metrics     │
                          └───┬──────────────┬────────┬──┘
                              │              │        │
                  enqueue     │              │ query  │ emit
                              ▼              │        ▼
                    ┌──────────────┐         │   ┌─────────────────┐
                    │ Redis (queue)│         │   │ OTel SDK        │
                    └──────┬───────┘         │   └────┬────────┬───┘
                           │                 │        │traces  │metrics
                    ┌──────▼───────┐         │        ▼        ▼
                    │Celery worker │         │    ┌───────┐ ┌──────────┐
                    │ PDF → chunks │         │    │ Tempo │ │Prometheus│
                    │ → embeddings │         │    └───┬───┘ └────┬─────┘
                    └──────┬───────┘         │        └────┬────┘
                           │                 │             ▼
                           ▼                 │        ┌─────────┐
                    ┌─────────────┐          │        │ Grafana │
                    │   Qdrant    │◄─────────┘        │  :3000  │
                    │   :6333     │  retrieve         └─────────┘
                    └─────────────┘

        ┌────────────────────────────────────────────────┐
        │  MLflow :5000                                  │
        │  • Experiment tracking (mọi eval run)          │
        │  • Prompt Registry (versioned prompts)         │
        │  • Model Registry (@champion / @challenger)    │
        └────────────────────────────────────────────────┘
                           ▲
                           │ đọc config champion khi khởi động
                    (FastAPI + Celery worker)
```

**Luồng dữ liệu chính:**

```
PDF upload → Celery → parse → chunk → embed → Qdrant
                                                  │
User question ──────────────────────────────────► retrieve → rerank → LLM → answer
                                                              │
                                                              └─► trace → Tempo
                                                                  metrics → Prometheus
```

---

## 2. Bảng phase tổng quan

| Phase | Chủ đề | Thời gian | Practice files | Deliverable ghép vào hệ thống |
|---|---|---|---|---|
| **0** | Nền móng & trả nợ kỹ thuật | 2 ngày | `practice/00_setup/` | `src/core/` — LLM + embedding provider chuẩn |
| **1** | Evaluation harness | 5–7 ngày | `practice/01_evaluation/` | `src/eval/` — chấm điểm tự động, log vào MLflow |
| **2** | Prompt optimization | 3–4 ngày | `practice/02_prompt_opt/` | `src/prompts/` — prompt registry + vòng lặp tối ưu |
| **3** | A/B testing | 5–7 ngày | `practice/03_ab_testing/` | `src/experiments/` — grid search cấu hình RAG |
| **4** | Serving: MLflow + FastAPI | 5–7 ngày | `practice/04_serving/` | `src/api/` — REST API + model registry |
| **5** | Observability | 5–7 ngày | `practice/05_observability/` | `src/telemetry/` — OTel + Prometheus + Grafana |
| **6** | Async với Celery | 3–4 ngày | `practice/06_async/` | `src/workers/` — ingest PDF ngầm |
| **7** | Ghép & deploy | 5–7 ngày | — | Toàn bộ hệ thống, `docker compose up` |

**Tổng: ~6–8 tuần** làm part-time (2–3h/ngày).

**Cách dùng lộ trình này:**
1. Đọc `docs/0X_*.md` của phase.
2. Chạy từng file trong `practice/0X_*/` theo thứ tự số. Mỗi file dạy **đúng một khái niệm**.
3. Làm phần **BÀI TẬP** cuối mỗi file.
4. Đọc mục "Ghép vào hệ thống" trong guide → tự viết code trong `src/`.
5. Check **Definition of Done** trước khi sang phase tiếp.

---

## 3. Chi tiết từng phase

### Phase 0 — Nền móng & trả nợ kỹ thuật `[2 ngày]`

> Không có phase này thì mọi con số ở phase sau đều là số rác.

**Học gì**
- Vì sao OpenRouter không làm được embedding, và ranh giới giữa "LLM provider" và "embedding provider".
- Embedding local với `sentence-transformers`: model chạy ở đâu, tốn RAM bao nhiêu, nhanh chậm thế nào.
- Tách **provider layer** ra khỏi business logic — điều kiện bắt buộc để A/B test được.

**Practice**

| File | Dạy gì |
|---|---|
| `00_env_check.py` | Kiểm tra env, API key, import — fail sớm và rõ ràng |
| `01_openrouter_hello.py` | Gọi OpenRouter trần bằng `httpx`, hiểu request/response thật |
| `02_local_embedding.py` | Load `bge-small-en-v1.5`, đo tốc độ, xem vector, tính cosine |
| `03_embedding_compare.py` | So 3 model embedding trên cùng cặp câu — thấy tận mắt vì sao phải A/B |

**Ghép vào hệ thống** → `src/core/providers.py`: `get_llm()`, `get_embedding(name)`, đọc từ `settings`.

**Definition of Done**
- [ ] `.venv` Python 3.11 chạy được, `pytest` pass
- [ ] Index build được **không gọi API embedding nào** (100% local)
- [ ] Đổi embedding model chỉ bằng sửa 1 dòng config
- [ ] Xoá singleton `_chatbot_instance`

---

### Phase 1 — Evaluation harness `[5–7 ngày]`

> Đây là phase quan trọng nhất cho CV. "Tôi build RAG" ai cũng nói được.
> "Tôi đo được RAG của tôi tốt lên 24%" thì rất ít người nói được.

**Học gì**
- **Hai tầng metric tách biệt** — hầu hết người mới gộp chung và đo sai:
  - *Retrieval metrics* (không cần LLM, rẻ, nhanh, deterministic): Hit Rate, MRR, NDCG@k, Context Precision/Recall.
  - *Generation metrics* (cần LLM-as-judge, đắt, có noise): Faithfulness, Answer Relevancy, Correctness.
- **LLM-as-judge** hoạt động ra sao: prompt chấm điểm, structured output, vì sao phải ép JSON schema, cách xử lý judge trả về rác.
- **Bias của judge**: position bias, verbosity bias, self-preference bias — và cách giảm.
- Vì sao phải **tách judge model khỏi generator model**.
- Golden dataset: bao nhiêu câu là đủ, cách sinh bán tự động, cách chống leak.

**Practice**

| File | Dạy gì |
|---|---|
| `01_llm_judge_basics.py` | Tự viết một judge từ đầu. Không thư viện. Hiểu tận gốc. |
| `02_structured_judge.py` | Ép judge trả JSON có schema, retry khi hỏng, tính độ tin cậy |
| `03_retrieval_metrics.py` | Tự cài Hit Rate / MRR / NDCG bằng numpy — 40 dòng, không magic |
| `04_generation_metrics.py` | Faithfulness + Answer Relevancy tự viết |
| `05_golden_dataset.py` | Sinh dataset Q&A từ chính documents bằng LLM + review thủ công |
| `06_mlflow_eval_run.py` | Log toàn bộ eval run vào MLflow: params, metrics, artifacts, table |
| `07_judge_reliability.py` | Đo độ ổn định của judge: chạy N lần, tính variance, so với người |

**Ghép vào hệ thống** → `src/eval/`:

```
src/eval/
  dataset.py      # load/validate golden set
  judges.py       # LLM-as-judge, có cache
  metrics.py      # retrieval metrics thuần numpy
  runner.py       # EvalRunner: config → chạy → log MLflow → trả EvalReport
```

**Definition of Done**
- [ ] Golden dataset ≥ **50 câu**, có `question / ground_truth / relevant_doc_ids`
- [ ] `python -m src.eval.runner --config baseline` chạy end-to-end và log vào MLflow
- [ ] Có ít nhất 5 metric, trong đó ≥2 metric không cần LLM
- [ ] Judge có **cache** — chạy lại không tốn tiền lần hai
- [ ] Chạy 2 lần cùng config → chênh lệch metric < 3% (đo được độ ổn định)

**Câu cho CV**: *"Xây dựng eval harness hai tầng (retrieval + generation) với LLM-as-judge có cache, chạy trên golden set 50 câu, độ lặp lại <3% variance."*

---

### Phase 2 — Prompt Optimization `[3–4 ngày]`

**Học gì**
- Prompt là **artifact có version**, không phải string trong code. MLflow Prompt Registry.
- Các kỹ thuật thực sự tạo khác biệt trên RAG: few-shot chọn động, chain-of-thought có kiểm soát, ép trích dẫn nguồn, xử lý "không biết", ràng buộc format.
- **Vòng lặp tối ưu tự động**: sinh biến thể → chấm bằng eval harness Phase 1 → giữ cái tốt nhất. Đây là mini-DSPy tự viết.
- Overfit prompt lên eval set — và cách chống bằng train/test split.

**Practice**

| File | Dạy gì |
|---|---|
| `01_prompt_registry.py` | Đăng ký prompt có version vào MLflow, load theo alias |
| `02_prompt_variants.py` | 5 biến thể system prompt, chấm cả 5 bằng harness Phase 1 |
| `03_few_shot_dynamic.py` | Chọn few-shot example theo similarity với câu hỏi |
| `04_citation_prompt.py` | Ép model trích dẫn `[doc_id]`, đo tỷ lệ trích dẫn đúng |
| `05_auto_optimize.py` | Vòng lặp: LLM tự sinh prompt mới từ lỗi của prompt cũ → chấm → lặp |

**Ghép vào hệ thống** → `src/prompts/registry.py` + `src/prompts/templates/`

**Definition of Done**
- [ ] Prompt lưu trong MLflow Prompt Registry, code load theo alias `@production`
- [ ] Có **bảng so sánh ≥5 biến thể** với số liệu
- [ ] Vòng auto-optimize chạy được ≥3 vòng và cải thiện được metric
- [ ] Tách train/test split cho prompt tuning — chứng minh không overfit

**Câu cho CV**: *"Tự động hoá prompt optimization bằng vòng lặp evaluate-mutate-select, nâng faithfulness từ 0.71 → 0.89 trên held-out test set."*

---

### Phase 3 — A/B Testing cấu hình RAG `[5–7 ngày]`

**Học gì**
- Thiết kế **experiment matrix**: chunker × chunk_size × embedding × retriever × top_k.
- Vì sao phải đổi **một biến tại một thời điểm** (OFAT) trước khi grid search.
- Retrieval strategies: dense, sparse (BM25), **hybrid + RRF**, reranking (cross-encoder).
- **Trade-off là điểm ăn tiền**: accuracy vs latency vs cost. Đây là thứ phân biệt AI Engineer với người chạy notebook.
- Ý nghĩa thống kê: chênh 2% trên 50 câu có thật không? Bootstrap confidence interval.

**Practice**

| File | Dạy gì |
|---|---|
| `01_chunker_ab.py` | 4 chunker × 3 chunk_size, đo retrieval metrics — 12 run |
| `02_embedding_ab.py` | 3 embedding model, cùng chunker — đo cả chất lượng và tốc độ |
| `03_retriever_ab.py` | dense vs BM25 vs hybrid-RRF vs +reranker |
| `04_grid_search.py` | Chạy toàn bộ matrix, log mỗi combo thành 1 MLflow run |
| `05_significance.py` | Bootstrap CI — trả lời "chênh lệch này có ý nghĩa không?" |
| `06_pareto_plot.py` | Vẽ Pareto front accuracy × latency × cost, chọn champion |

**Ghép vào hệ thống** → `src/experiments/`:

```
src/experiments/
  space.py        # định nghĩa search space
  runner.py       # chạy 1 combo → EvalReport
  sweep.py        # chạy N combo, song song, log MLflow
```

**Definition of Done**
- [ ] ≥ **20 MLflow run** với params đầy đủ, so sánh được trên UI
- [ ] Bảng kết quả có cả **accuracy, p95 latency, cost/query**
- [ ] Có biểu đồ Pareto và **lý do chọn champion bằng chữ**
- [ ] Bootstrap CI cho khác biệt giữa champion và baseline

**Câu cho CV**: *"Thực hiện 24 thí nghiệm A/B trên chunking/embedding/retrieval; hybrid search + cross-encoder rerank nâng NDCG@5 từ 0.62 → 0.81, đánh đổi +180ms p95."*

---

### Phase 4 — Serving: MLflow Registry + FastAPI `[5–7 ngày]`

**Học gì**
- **MLflow `pyfunc`**: đóng gói cả pipeline RAG (không chỉ model) thành artifact tái tạo được.
- **Model Registry + alias**: `@champion` / `@challenger`. Deploy = đổi alias, không sửa code.
- FastAPI production-grade: lifespan, dependency injection, Pydantic schema, streaming SSE, error handling.
- Vì sao API cần `/health` và `/ready` **khác nhau**.
- Load model lúc startup, không phải mỗi request.

**Practice**

| File | Dạy gì |
|---|---|
| `01_pyfunc_wrapper.py` | Wrap RAG pipeline thành `mlflow.pyfunc.PythonModel` |
| `02_log_and_register.py` | Log model + signature + requirements, đăng ký alias `@champion` |
| `03_load_from_registry.py` | Load theo alias, verify signature, đo cold-start |
| `04_fastapi_minimal.py` | FastAPI với `/health`, `/query` — bản nhỏ nhất chạy được |
| `05_fastapi_streaming.py` | SSE streaming token — UX thật cần cái này |
| `06_fastapi_full.py` | Lifespan load model, DI, error handler, request ID |

**Ghép vào hệ thống** → `src/api/`:

```
src/api/
  main.py         # app + lifespan
  routes/         # query.py, ingest.py, health.py
  schemas.py      # Pydantic request/response
  deps.py         # dependency injection
```

**Definition of Done**
- [ ] `POST /query` trả lời được, có `sources[]` và `latency_ms`
- [ ] Streaming hoạt động (thấy token chảy ra)
- [ ] Đổi champion trong MLflow → restart API → hành vi đổi, **không sửa dòng code nào**
- [ ] `/health` (liveness) và `/ready` (đã load model chưa) tách biệt
- [ ] Docker image build được

**Câu cho CV**: *"Đóng gói RAG pipeline thành MLflow pyfunc, serve qua FastAPI với streaming SSE; promote model bằng alias registry — deploy không cần đổi code."*

---

### Phase 5 — Observability `[5–7 ngày]`

> Phase khiến CV nhìn như của người đã đi làm, không phải người mới học.

**Học gì**
- **Ba trụ cột**: metrics (Prometheus), traces (OTel → Tempo), logs (structured JSON). Khác nhau ở đâu, khi nào dùng cái nào.
- OpenTelemetry: span, trace context propagation, attribute, semantic convention. Auto-instrument vs manual span.
- **Metric nào thực sự quan trọng cho LLM app** — không phải CPU/RAM:
  - `rag_query_duration_seconds` (histogram, tách theo stage: retrieve / rerank / generate)
  - `llm_tokens_total{type=prompt|completion}` → suy ra **cost thật**
  - `retrieval_top_score` (distribution) → phát hiện khi retrieval xấu đi
  - `rag_errors_total{stage,type}`
  - `rag_no_answer_total` → tỷ lệ "tôi không biết"
- Histogram vs Summary, chọn bucket sao cho đo được p95 đúng.
- Grafana dashboard + alert rule.
- Trace một query xuyên FastAPI → Celery → Qdrant → OpenRouter.

**Practice**

| File | Dạy gì |
|---|---|
| `01_prometheus_basics.py` | Counter/Gauge/Histogram, expose `/metrics`, scrape thử |
| `02_histogram_buckets.py` | Vì sao bucket sai làm p95 sai — demo bằng số |
| `03_otel_hello.py` | Span thủ công, xem trace trong console exporter |
| `04_otel_to_tempo.py` | Gửi trace qua OTLP đến Tempo, xem trên Grafana |
| `05_instrument_rag.py` | Gắn span cho từng stage của RAG, gắn attribute (model, top_k, tokens) |
| `06_llm_cost_metric.py` | Đếm token → tính $/query → expose thành metric |
| `07_structured_logging.py` | JSON log có `trace_id` — nối log với trace |

**Ghép vào hệ thống** → `src/telemetry/`:

```
src/telemetry/
  otel.py         # setup tracer provider, OTLP exporter
  metrics.py      # định nghĩa toàn bộ Prometheus metric
  logging.py      # structlog JSON + trace correlation
  middleware.py   # FastAPI middleware tự động
```

+ `infra/grafana/dashboards/rag.json` — dashboard commit vào repo.

**Definition of Done**
- [ ] `docker compose up` → Grafana có dashboard với ≥6 panel
- [ ] Trace một query hiện đủ span: `retrieve → rerank → generate`
- [ ] Dashboard hiện **cost tích luỹ theo thời gian**
- [ ] Có ≥2 alert rule (p95 latency > 5s, error rate > 5%)
- [ ] Log có `trace_id`, click từ Grafana log → trace được

**Câu cho CV**: *"Thiết lập observability đầy đủ 3 trụ cột với OpenTelemetry + Prometheus + Grafana + Tempo; dashboard theo dõi p95 latency theo từng stage, token cost real-time và retrieval quality drift."*

---

### Phase 6 — Async processing với Celery `[3–4 ngày]`

**Học gì**
- Vì sao ingest PDF **phải** async: parse + embed 200 trang mất 60–300s, HTTP request sẽ timeout.
- Celery: broker vs result backend, task routing, retry với exponential backoff, idempotency.
- Progress reporting: `update_state` → client poll `/jobs/{id}`.
- Task nào nên async, task nào không (query thì **không** — user đang đợi).
- Trace context propagation qua ranh giới process — không tự động, phải làm tay.

**Practice**

| File | Dạy gì |
|---|---|
| `01_celery_hello.py` | Task đầu tiên, chạy worker, gọi `.delay()` |
| `02_celery_progress.py` | Long task báo tiến độ, client poll trạng thái |
| `03_celery_retry.py` | Retry khi OpenRouter 429, backoff, dead-letter |
| `04_celery_ingest_pdf.py` | Task thật: PDF → chunk → embed → Qdrant, có progress |
| `05_celery_otel.py` | Truyền trace context từ API sang worker |

**Ghép vào hệ thống** → `src/workers/`:

```
src/workers/
  app.py          # Celery app config
  tasks/ingest.py # PDF ingestion pipeline
  tasks/eval.py   # chạy eval theo lịch (nightly)
```

**Definition of Done**
- [ ] `POST /ingest` trả `job_id` trong <200ms
- [ ] `GET /jobs/{id}` trả progress %
- [ ] Worker crash giữa chừng → task retry, không mất dữ liệu
- [ ] Trace chạy xuyên từ API sang worker (một trace_id duy nhất)
- [ ] Nightly eval job chạy tự động, log vào MLflow

**Câu cho CV**: *"Xử lý ingestion PDF bất đồng bộ bằng Celery + Redis với progress tracking và retry backoff; API trả về trong <200ms thay vì block 3 phút."*

---

### Phase 7 — Ghép & Deploy `[5–7 ngày]`

**Học gì**
- PDF parsing thật: layout, bảng, header/footer, OCR khi cần.
- Multi-tenancy: mỗi user/document một collection Qdrant, tách biệt dữ liệu.
- Docker Compose orchestration, healthcheck, thứ tự khởi động, volume.
- Bảo mật tối thiểu: API key auth, rate limit, giới hạn size upload, validate file type.
- CI: GitHub Actions chạy test + eval regression trên mỗi PR.

**Việc phải làm**
1. `src/ingestion/pdf.py` — parser thật (PyMuPDF), giữ metadata trang
2. Multi-tenant collection trong Qdrant
3. `docker-compose.yml` đầy đủ service + healthcheck
4. Streamlit UI: upload PDF → thấy progress → chat
5. `Makefile`: `make up`, `make eval`, `make sweep`, `make test`
6. GitHub Actions: test + **eval regression gate** (metric tụt >5% thì fail PR)
7. `README.md` có kiến trúc, số liệu, GIF demo

**Definition of Done**
- [ ] Người lạ clone repo → `docker compose up` → dùng được, không hỏi gì thêm
- [ ] Upload PDF 100+ trang → hỏi đáp đúng
- [ ] README có **bảng số liệu A/B thật** và ảnh dashboard
- [ ] CI xanh, có eval gate
- [ ] Có `docs/DECISIONS.md` ghi lý do từng lựa chọn kiến trúc

---

## 4. Thứ tự ưu tiên nếu thiếu thời gian

Nếu chỉ có 3 tuần thay vì 8, làm theo thứ tự này — mỗi phase vẫn tự đứng được:

1. **Phase 0 + 1** (bắt buộc — không có eval thì mọi thứ sau vô nghĩa)
2. **Phase 3** (A/B testing — ấn tượng nhất trên CV, dùng lại harness Phase 1)
3. **Phase 4** (có API = có sản phẩm)
4. **Phase 5** (observability — điểm khác biệt)
5. Phase 2, 6, 7

---

## 5. Rủi ro và cách xử lý

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Đốt tiền OpenRouter khi eval | Bill tăng nhanh | Cache judge response ra đĩa (Phase 1 đã có); dùng model rẻ làm judge |
| Rate limit 429 | Eval fail giữa chừng | Retry backoff + giới hạn concurrency |
| Torch cài nặng/lỗi trên Windows | `pip install` fail | Cài CPU-only wheel: `--index-url https://download.pytorch.org/whl/cpu` |
| Docker ăn hết RAM | Máy đơ | Chạy từng nhóm service; Qdrant + Redis nhẹ, Tempo/Grafana nặng hơn |
| Overfit lên eval set | Metric đẹp, thực tế tệ | Train/test split từ Phase 2; giữ 20% câu không bao giờ dùng để tune |
| Sa lầy hoàn hảo hoá | Một phase quá 2× thời gian dự kiến | Chốt Definition of Done, sang phase sau, quay lại nếu còn thời gian |

---

## 6. Bảng kết quả cần điền (in ra, dán lên tường)

Đây là thứ cuối cùng đi vào CV. Điền dần qua các phase.

| Chỉ số | Baseline (Phase 0) | Champion (Phase 3) | Cải thiện |
|---|---|---|---|
| Hit Rate @5 | | | |
| NDCG @5 | | | |
| Faithfulness | | | |
| Answer Relevancy | | | |
| p50 latency | | | |
| p95 latency | | | |
| Cost / 1000 query | | | |
| Index build time | | | |

---

## 7. File hướng dẫn chi tiết

| Phase | Guide |
|---|---|
| 0 | [`docs/00_setup.md`](docs/00_setup.md) |
| 1 | [`docs/01_evaluation.md`](docs/01_evaluation.md) |
| 2 | [`docs/02_prompt_optimization.md`](docs/02_prompt_optimization.md) |
| 3 | [`docs/03_ab_testing.md`](docs/03_ab_testing.md) |
| 4 | [`docs/04_serving.md`](docs/04_serving.md) |
| 5 | [`docs/05_observability.md`](docs/05_observability.md) |
| 6 | [`docs/06_async_celery.md`](docs/06_async_celery.md) |
| 7 | [`docs/07_production.md`](docs/07_production.md) |
| — | [`docs/CV_TALKING_POINTS.md`](docs/CV_TALKING_POINTS.md) — cách kể project khi phỏng vấn |
