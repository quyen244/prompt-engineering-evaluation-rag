# Phase 5 — Observability

> Phase khiến CV nhìn như của người đã đi làm, không phải người mới học.
> Rất ít ứng viên junior nói được về cardinality, bucket, hay fail-open.

---

## Mục tiêu

1. **Ba trụ cột** chạy được và **nối được với nhau**: metric → trace → log.
2. Bộ metric **đúng cho ứng dụng LLM** — không phải CPU/RAM.
3. Histogram có bucket chọn theo phân bố thật, p95 đo đúng.
4. Trace xuyên stage, có attribute theo semantic convention `gen_ai.*`.
5. Metric chi phí, alert theo **tốc độ chi tiền**.
6. Log JSON có `trace_id`, đi từ biểu đồ tới đúng dòng log trong 3 cú click.

---

## Nền tảng lý thuyết

### 1. Ba trụ cột trả lời ba câu hỏi khác nhau

```
   METRIC              TRACE                    LOG
   "có bất thường      "request NÀY đi qua       "chính xác chuyện gì
    không?"             đâu, chậm ở đâu?"        đã xảy ra?"

   rẻ nhất             đắt vừa                  đắt nhất
   tổng hợp            theo từng request        theo từng sự kiện
   ALERT ĐƯỢC          không alert được         không alert được
   giữ 1 năm           giữ 7-30 ngày            giữ 3-14 ngày
```

Quy trình điều tra thật luôn theo thứ tự đó:

1. Alert kêu (**metric**): p95 vượt ngưỡng.
2. Mở Grafana → Tempo, tìm trace chậm nhất (**trace**): thấy stage `generate`
   mất 22 giây.
3. Bấm "Logs for this span" (**log**): đọc thấy `error_type: TimeoutError`,
   `prompt_tokens: 8400`.

Bước 3 chỉ chạy được nếu log **có `trace_id`**, và phải là id **lấy từ OTel** —
không phải một id bạn tự sinh riêng cho log.

### 2. Đo cái gì — CPU/RAM là vô dụng

Ứng dụng LLM nhàn rỗi 99% thời gian vì chờ mạng. CPU 4% không nói lên điều gì.
Thứ cần đo:

| Metric | Kiểu | Vì sao |
|---|---|---|
| `rag_query_duration_seconds{stage}` | Histogram | **Tách theo stage** — điểm mấu chốt |
| `rag_retrieval_top_score` | Histogram | Chất lượng, không phải tốc độ. Tụt trước khi có ai phàn nàn |
| `rag_no_answer_total` | Counter | Tỉ lệ "không tìm thấy" — tín hiệu sản phẩm |
| `rag_errors_total{stage,type}` | Counter | Hệ thống hỏng thật |
| `llm_tokens_total{type,model}` | Counter | Suy ra chi phí |
| `llm_cost_usd_total{model}` | Counter | `rate()` → $/giờ, alert được |
| `rag_context_tokens` | Histogram | Dự báo chi phí tốt hơn đếm request |
| `rag_queries_in_flight` | Gauge | Đo tải đồng thời |

**Vì sao phải tách theo stage?** p95 tổng là 4.2 giây không nói bạn nên sửa gì.
Tách ra: retrieve 62ms, generate 812ms → generate chiếm 93%. Kết luận ngay: tối
ưu chunking sẽ không cứu được latency; phải động vào tầng generate.

**`rag_errors_total` ≠ `rag_no_answer_total`.** Cái đầu là hệ thống hỏng — alert,
đánh thức người trực. Cái sau là hệ thống chạy **đúng** nhưng tài liệu không có
câu trả lời — không phải sự cố, nhưng tỉ lệ tăng đột ngột là tín hiệu rất mạnh.
Phân biệt được hai thứ này là dấu hiệu của người đã vận hành RAG thật.

### 3. Cardinality — cách làm nổ Prometheus

Mỗi **tổ hợp label** là một chuỗi thời gian riêng, tốn RAM vĩnh viễn.

| Dữ liệu | Số giá trị | Prometheus label? | Span attribute? |
|---|---|---|---|
| `stage` | 3 | ✓ nên dùng | ✓ |
| `model` | ~5 | ✓ | ✓ |
| `error_type` | ~15 | ✓ | ✓ |
| `user_id` | hàng triệu | **✗ NỔ** | ✓ thoải mái |
| `question` | vô hạn | **✗ NỔ** | ✓ (nhưng xem mục 6) |
| `chunk_ids` | tổ hợp | **✗ NỔ** | ✓ |

Quy tắc: **Prometheus = "bao nhiêu, nhanh cỡ nào" (cardinality thấp). Trace =
"chuyện gì đã xảy ra với request NÀY" (tự do).**

Span không có tích Descartes — mỗi span là một bản ghi độc lập. Đó là lý do
nhét `user_id` vào span thì ổn còn nhét vào label thì giết Prometheus.

### 4. Bucket sai làm p95 sai

Prometheus không lưu từng giá trị. Nó đếm số quan sát rơi vào từng bucket, rồi
`histogram_quantile()` **nội suy tuyến tính** trong bucket chứa phân vị.

Hệ quả: nếu 95% giá trị rơi vào một bucket, p95 là một con số bịa.

```
Bucket mặc định của prometheus_client:
  .005 .01 .025 .05 .075 .1 .25 .5 .75 1.0 2.5 5.0 7.5 10.0

Latency LLM thật: 0.8 – 25 giây
→ gần như mọi giá trị rơi vào bucket [10, +Inf]
→ p95 nội suy trong bucket vô hạn → SAI HOÀN TOÀN
```

Và retrieval (20-90ms) với generation (0.3-15s) lệch nhau hai bậc độ lớn — **không
dùng chung bộ bucket được**. `02_histogram_buckets.py` chứng minh bằng số.

Cách chọn: chạy hệ thống, thu latency thật, đặt bucket dày quanh vùng p50-p99.

### 5. Fail open — telemetry không được giết sản phẩm

```
hệ thống quan sát hỏng  →  hệ thống sản phẩm VẪN CHẠY
```

Nghe hiển nhiên, rất dễ vi phạm:

```python
span.set_attribute("cost", compute_cost(response))   # ← compute_cost ném lỗi
                                                     #   → chết request của user
```

Hai hệ quả cụ thể:

- **`BatchSpanProcessor`, không phải `SimpleSpanProcessor`.** Simple gửi đồng bộ,
  chặn request. Batch gom hàng đợi, gửi ở luồng nền, và **vứt span khi đầy** —
  đó là quyết định có ý thức, không phải bug.
- **Mọi tính toán cho telemetry phải bọc `try/except`,** hoặc đơn giản tới mức
  không thể lỗi.

Nhưng phải **biết** khi đang mất: đưa `otel_span_processor_dropped_spans` lên
dashboard.

### 6. Chi phí — loại sự cố riêng của LLM app

Hệ thống chạy hoàn hảo, không lỗi, latency đẹp — và hoá đơn gấp 40 lần.

RAG có hình dạng chi phí đặc trưng: **prompt token thống trị**. Context 5 chunk
× 200 token = 1000 vào, câu trả lời 150 ra. Tỉ lệ vào/ra ~7:1.

Ai đó đổi `top_k` từ 5 lên 20 → chi phí tăng ~3.7 lần, latency gần như không đổi,
không metric nào kêu. Bạn phát hiện sau 30 ngày.

**Alert theo tốc độ, không theo tổng:**

```yaml
- alert: LLMCostSpike
  expr: sum(rate(llm_cost_usd_total[15m])) * 3600 > 2.0   # USD/giờ
  for: 10m

- alert: ContextTokensGrowing        # bắt NGUYÊN NHÂN, không phải hậu quả
  expr: histogram_quantile(0.95, sum(rate(rag_context_tokens_bucket[1h])) by (le)) > 4000
  for: 30m
```

"Đã tiêu $500" → biết lúc đã muộn. "Đang tiêu $9/giờ" → cứu kịp. Đó là lý do chi
phí phải là **Counter** (chỉ tăng): `rate()` không dùng được trên Gauge.

### 7. Log: JSON, có context, và biết kiêng

- **JSON thay text.** `duration_ms > 5000` là truy vấn; regex trên chuỗi là nợ.
- **`contextvars`** để `trace_id`/`request_id` bám vào mọi log line mà không phải
  thêm tham số vào chữ ký của 8 tầng hàm. An toàn với cả thread lẫn asyncio.
- **Đừng log thứ không được log:**

| Trường | Nên | Vì sao |
|---|---|---|
| `question` (nội dung) | ✗ hash + độ dài | Có thể chứa dữ liệu cá nhân |
| `answer` (nội dung) | ✗ chỉ độ dài | Chứa lại nội dung tài liệu; log lưu lâu hơn |
| `api_key` | ✗ **TUYỆT ĐỐI KHÔNG** | Log được sao chép đi nhiều nơi, lưu nhiều năm |
| `chunk_ids`, `top_score`, token counts | ✓ | Định danh và số thuần, cần để debug |

Log rò rỉ là sự cố khó sửa nhất — log đã sang backup, sang công cụ phân tích.
"Xoá đi" không còn nghĩa gì.

---

## Thứ tự chạy practice

| # | File | Học gì | Cần gì |
|---|---|---|---|
| 01 | `01_prometheus_basics.py` | Counter/Gauge/Histogram/Summary, `/metrics`, cardinality | `prometheus_client` |
| 02 | `02_histogram_buckets.py` | Bucket sai → p95 sai, chứng minh bằng số | numpy |
| 03 | `03_otel_hello.py` | Span, trace, attribute, event — ConsoleExporter | `opentelemetry-sdk` |
| 04 | `04_otel_to_tempo.py` | OTLP, Resource, Batch vs Simple, fail open, `shutdown()` | + exporter-otlp-proto-http |
| 05 | `05_instrument_rag.py` | Decorator `@traced`, metric nào quan trọng, attribute vs label | cả hai |
| 06 | `06_llm_cost_metric.py` | Token → tiền → metric → alert rule | `prometheus_client` |
| 07 | `07_structured_logging.py` | JSON log, contextvars, `trace_id`, cái gì không được log | `structlog` (tuỳ chọn) |

```powershell
pip install prometheus_client opentelemetry-sdk opentelemetry-exporter-otlp-proto-http structlog

python practice/05_observability/01_prometheus_basics.py
python practice/05_observability/02_histogram_buckets.py
python practice/05_observability/03_otel_hello.py
python practice/05_observability/04_otel_to_tempo.py
python practice/05_observability/05_instrument_rag.py
python practice/05_observability/06_llm_cost_metric.py
python practice/05_observability/07_structured_logging.py | jq
```

File 04 chạy được **không cần Tempo** — exporter thất bại êm ái, và đó chính là
demo 4 của nó.

---

## Ghép vào hệ thống

### Cấu trúc cần tạo

```
src/telemetry/
  __init__.py
  otel.py          ← setup_tracing(): TracerProvider + Resource + OTLP exporter
  metrics.py       ← TOÀN BỘ Prometheus metric, định nghĩa MỘT LẦN
  logging.py       ← structlog JSON + contextvars + trace correlation
  decorators.py    ← @traced(stage)
  middleware.py    ← FastAPI middleware: request_id, bind context, đếm metric
```

### Signature nên có

```python
# src/telemetry/otel.py
def setup_tracing(service_name: str, endpoint: str | None = None) -> TracerProvider: ...
def shutdown_tracing() -> None: ...      # gọi trong lifespan shutdown

# src/telemetry/metrics.py
RAG_DURATION: Histogram      # labels: stage
RAG_TOP_SCORE: Histogram
RAG_ERRORS: Counter          # labels: stage, type
RAG_NO_ANSWER: Counter
RAG_IN_FLIGHT: Gauge
LLM_TOKENS: Counter          # labels: type, model
LLM_COST: Counter            # labels: model, endpoint
CONTEXT_TOKENS: Histogram

def record_llm_call(model: str, prompt_tokens: int, completion_tokens: int,
                    endpoint: str = "query") -> float: ...

# src/telemetry/decorators.py
def traced(stage: str) -> Callable: ...

# src/telemetry/logging.py
def setup_logging(json: bool = True) -> None: ...
def bind(**kwargs) -> None: ...
def clear_context() -> None: ...
```

### Nguyên tắc thiết kế phải giữ

1. **Metric định nghĩa MỘT LẦN ở module level.** Tạo lại metric cùng tên ném
   `ValueError`. Đó là lý do chúng phải nằm trong `metrics.py` và được import,
   không bao giờ tạo bên trong hàm.
2. **Đo trong `finally`.** Request lỗi sau 30 giây timeout mà không vào histogram
   sẽ khiến p95 đẹp một cách giả tạo — bạn chỉ đo request **thành công**. Lỗi
   tinh vi và rất phổ biến.
3. **Bucket riêng cho mỗi stage** nếu phân bố lệch nhau bậc độ lớn.
4. **`shutdown()` provider trong lifespan shutdown.** Không có nó, span cuối cùng
   nằm trong hàng đợi và biến mất cùng process. Đây là nguyên nhân số một của
   "tôi instrument rồi mà Tempo trống trơn".
5. **Telemetry tách khỏi logic nghiệp vụ.** Decorator, middleware — không rải
   `metric.inc()` khắp code RAG.
6. **`/metrics` phải nằm ngoài auth** (Prometheus không đăng nhập được) nhưng
   **trong mạng nội bộ**.

### Sửa code cũ

| File | Việc cần làm |
|---|---|
| `src/api/main.py` | Thêm `setup_tracing()` + `setup_logging()` vào lifespan; `shutdown_tracing()` khi tắt |
| `src/api/main.py` | Middleware: sinh `request_id`, `bind()` context, `clear_context()` ở cuối |
| `src/api/routes/*` | Mount `/metrics` bằng `prometheus_client.make_asgi_app()` |
| `src/rag/*` | Gắn `@traced("retrieve")`, `@traced("generate")` — không sửa gì khác |
| `src/core/providers.py` | Gọi `record_llm_call()` sau mỗi lần gọi LLM |

### Infra

```
infra/prometheus/prometheus.yml     scrape config, trỏ vào API:8000/metrics
infra/prometheus/alerts.yml         LLMCostSpike, ContextTokensGrowing, HighErrorRate
infra/tempo/tempo.yaml              nhận OTLP ở 4317/4318
infra/grafana/provisioning/datasources/datasources.yml
infra/grafana/dashboards/rag.json   COMMIT VÀO REPO
```

Dashboard commit vào repo là chi tiết nhỏ nhưng gây ấn tượng: nó chứng minh bạn
coi observability là **code**, không phải thứ ai đó click tay rồi mất khi container
bị xoá.

---

## Bẫy thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Tempo trống trơn | Không gọi `provider.shutdown()` | Flush trước khi thoát; trong lifespan shutdown |
| `ValueError: Duplicated timeseries` | Định nghĩa metric hai lần | Đưa vào `metrics.py`, import |
| p95 sai lệch hoàn toàn | Dùng bucket mặc định cho latency LLM | Bucket theo phân bố thật (file 02) |
| Prometheus ngốn hết RAM | Label cardinality cao | Bỏ `user_id`/`question` khỏi label |
| Span có exception nhưng Tempo tô xanh | Thiếu `set_status(StatusCode.ERROR)` | `record_exception()` **và** `set_status()` |
| p95 đẹp bất thường | Chỉ đo request thành công | Đo trong `finally` |
| Trace bị đứt ở Celery | Không truyền trace context qua process | Phase 6 — header `traceparent` |
| `service.name = unknown_service` | Thiếu Resource | `Resource.create({"service.name": ...})` |
| API chậm hẳn sau khi thêm OTel | Dùng `SimpleSpanProcessor` | Đổi sang `BatchSpanProcessor` |
| Log không có `trace_id` | Tự sinh id riêng cho log | Lấy từ `trace.get_current_span()` |
| Log của request A lẫn sang B | Không `clear_context()` | Clear ở cuối middleware |
| OTLP 404 | Thiếu `/v1/traces` trong endpoint HTTP | `{endpoint}/v1/traces` |

---

## Definition of Done

- [ ] `/metrics` chạy, Prometheus scrape được, thấy metric trong UI.
- [ ] Histogram bucket chọn theo latency **thật của bạn**, không phải mặc định.
- [ ] Trace hiện trong Tempo, có cây span `rag.query → retrieve → generate`.
- [ ] Span có attribute `gen_ai.request.model`, `gen_ai.usage.*`, `rag.top_score`.
- [ ] Grafana dashboard có ≥ 6 panel: p50/p95 theo stage, error rate, token/phút,
      $/giờ, top_score distribution, no-answer rate.
- [ ] Dashboard JSON **commit trong repo**, không phải click tay.
- [ ] ≥ 2 alert rule chạy được, tự tay kích hoạt thử một cái.
- [ ] Log JSON, có `trace_id`, không chứa nội dung câu hỏi.
- [ ] **Đi được từ một điểm nhọn trên biểu đồ p95 → trace → dòng log tương ứng.**
      Đây là bài kiểm tra thật của cả phase.
- [ ] Tắt Tempo → API vẫn chạy bình thường (fail open).

**Câu cho CV** (điền số thật):

> Xây observability đầy đủ ba trụ cột cho hệ thống RAG: Prometheus (histogram
> theo từng stage với bucket hiệu chỉnh, metric chi phí LLM), OpenTelemetry →
> Tempo (trace xuyên FastAPI/Celery), structlog JSON tương quan qua `trace_id`.
> Dashboard và alert rule quản lý bằng code. Phát hiện được __% thời gian nằm ở
> tầng generate và cắt chi phí __% bằng alert vào context token.

---

## Câu hỏi phỏng vấn

**1. Metric, trace, log khác nhau thế nào — khi nào dùng cái nào?**
Metric trả lời "có bất thường không", rẻ và alert được. Trace trả lời "request
này chậm ở đâu". Log trả lời "chính xác chuyện gì đã xảy ra". Điều tra luôn đi
theo thứ tự đó, và bước cuối chỉ chạy được nếu log có `trace_id` lấy từ OTel.

**2. Bạn đo gì cho ứng dụng LLM?**
Không phải CPU/RAM — chúng nhàn rỗi vì 99% thời gian là chờ mạng. Tôi đo latency
**tách theo stage**, phân bố `top_score` của retrieval, tỉ lệ no-answer, error
theo stage và loại, và token → chi phí. Tách theo stage là quan trọng nhất: p95
tổng không nói tôi nên sửa gì.

**3. Vì sao không cho `user_id` vào Prometheus label?**
Mỗi tổ hợp label là một chuỗi thời gian tốn RAM vĩnh viễn. Một triệu user × 10
stage = 10 triệu chuỗi, Prometheus chết. `user_id` thuộc về span attribute — span
không có tích Descartes, mỗi cái là bản ghi độc lập.

**4. Histogram bucket chọn thế nào?**
Theo phân bố thật, đo trước rồi mới chọn. Bucket mặc định của client dừng ở 10
giây, mà latency LLM là 0.8-25 giây, nên gần như mọi giá trị rơi vào bucket vô
hạn và `histogram_quantile` nội suy ra số bịa. Retrieval và generation lệch nhau
hai bậc độ lớn nên phải có bộ bucket riêng.

**5. Collector chết thì ứng dụng sao?**
Vẫn chạy. Đó là nguyên tắc fail-open: telemetry hỏng không được kéo theo sản
phẩm hỏng. Cụ thể là dùng `BatchSpanProcessor` gửi ở luồng nền và vứt span khi
hàng đợi đầy, cộng với việc bọc `try/except` quanh mọi tính toán cho telemetry.

**6. Kiểm soát chi phí LLM thế nào?**
Đếm token thành Counter, quy ra USD với đơn giá in/out riêng, rồi alert theo
`rate()` — USD/giờ, không phải tổng tích luỹ. Tôi còn alert vào p95 của
`rag_context_tokens`, tức là bắt **nguyên nhân** (context phình to vì ai đó tăng
`top_k`) thay vì hậu quả.

**7. p95 của bạn có thể nói dối ở chỗ nào?**
Ba chỗ. Bucket sai (nội suy trong bucket vô hạn). Chỉ đo request thành công —
phải đo trong `finally`, nếu không timeout 30 giây bị bỏ ngoài. Và đo sai vị trí
với streaming: nếu chốt thời gian trước khi generator chạy xong thì bạn đang đo
TTFT mà tưởng là tổng.

**8. Vì sao `rag_no_answer_total` không phải là error?**
Vì hệ thống chạy đúng — tài liệu thật sự không có câu trả lời. Alert vào nó sẽ
đánh thức người trực vì một chuyện không phải sự cố. Nhưng tỉ lệ tăng đột ngột
là tín hiệu sản phẩm rất mạnh: corpus thiếu, retrieval xấu đi, hoặc user đang
hỏi về chủ đề mới.

---

← [Phase 4 — Serving](04_serving.md) · [Phase 6 — Async & Celery](06_async_celery.md) →
