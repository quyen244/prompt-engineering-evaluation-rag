# Phase 4 — Serving: MLflow Registry + FastAPI

> Hết phase này bạn có **sản phẩm**, không còn là script. Có API thì mới có
> gì để observe ở Phase 5, mới có gì để đưa vào Docker ở Phase 7.

---

## Mục tiêu

1. RAG pipeline đóng gói thành **MLflow pyfunc model** có signature và requirements.
2. Model **đăng ký vào registry**, deploy bằng **alias** `@champion`.
3. **FastAPI** với `/health`, `/ready`, `/query` — model load lúc startup.
4. **SSE streaming** — token chảy ra thay vì chờ 6 giây màn hình trắng.
5. Request ID, exception handler tập trung, phân loại mã lỗi đúng.
6. Đổi champion → restart → hành vi đổi, **không sửa dòng code nào**.

---

## Nền tảng lý thuyết

### 1. Vì sao pyfunc chứ không phải flavor có sẵn

MLflow có sẵn flavor cho sklearn, pytorch, transformers... nhưng RAG không
phải một model. Nó là một **pipeline**: embedding model + index + prompt +
LLM ở xa + logic ghép chúng lại.

`mlflow.pyfunc.PythonModel` cho bạn định nghĩa tự do `predict()` làm gì.
Đổi lại, bạn phải tự lo 4 thứ:

```python
class RagModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config):        # chỉ giữ CẤU HÌNH
        self.config = config           # mọi thứ ở đây đều bị pickle

    def load_context(self, context):   # chạy 1 lần lúc LOAD
        corpus = context.artifacts["corpus"]   # đường dẫn KHÁC lúc log
        self.pipeline = build(self.config, corpus)

    def predict(self, context, model_input, params=None):
        ...                            # model_input là DataFrame
```

**Ranh giới `__init__` / `load_context` là bài học lớn nhất của phase này.**

| | `__init__` | `load_context` |
|---|---|---|
| Chạy khi nào | Lúc bạn tạo object để log | Lúc MLflow load model ra để serve |
| Chứa gì | Cấu hình (kiểu nguyên thuỷ) | Trạng thái nặng: model, index |
| Bị pickle không | **Có** | Không |

Đặt nhầm việc nặng vào `__init__` → artifact phình lên hàng trăm MB, và tệ hơn:
nó chứa index **cũ**, không build lại được khi corpus đổi.

Kiểm tra nhanh: `python_model.pkl` trong artifact phải là **vài KB**.

### 2. Alias, không phải stage

`transition_model_version_stage` (Staging/Production/Archived) đã **deprecated
từ MLflow 2.9**. Đừng dùng, và đừng viết vào CV.

```python
client.set_registered_model_alias("rag-pdf-qa", "champion", "3")
model_uri = "models:/rag-pdf-qa@champion"
```

Ba lý do alias thắng stage:

- Chỉ có 4 stage cố định — không diễn tả nổi canary, shadow, champion-per-tenant.
- Một version chỉ ở được **một** stage — không thể vừa Production cho khách A
  vừa Staging cho khách B.
- Tên stage không nói lên ngữ nghĩa: "Production" của team nào?

**Dùng alias ở đâu, dùng version number ở đâu:**

| Tình huống | Dùng gì | Vì sao |
|---|---|---|
| API production | `@champion` | Deploy = đổi alias |
| Job eval hằng đêm | `@champion` + `@challenger` | So sánh liên tục |
| **Điều tra sự cố** | **số version** | Alias có thể đã đổi rồi — load `@champion` sẽ ra model khác với cái gây sự cố |

Hệ quả: **luôn trả `model_version` trong response** và log nó ở mỗi request.

### 3. `/health` ≠ `/ready` — và gộp chúng là lỗi nghiêm trọng

```
/health (liveness)   "process còn sống không?"
                     FAIL → orchestrator GIẾT container và tạo cái mới

/ready  (readiness)  "nhận traffic được chưa?"
                     FAIL → load balancer NGỪNG gửi traffic, container VẪN SỐNG
```

Gộp làm một gây một trong hai thảm hoạ:

- Dùng logic "đã load model chưa" cho **liveness** → container bị giết **trong
  lúc đang load model**, khởi động lại, lại bị giết. Vòng lặp chết vĩnh viễn.
- Dùng logic "process còn sống" cho **readiness** → traffic được gửi vào lúc
  model chưa load xong → 500 hàng loạt ở mỗi lần deploy.

Vì cold-start của RAG là 10-30 giây, đây không phải chuyện lý thuyết.

### 4. Cold-start quyết định kiến trúc deploy

| Cold-start | Hệ quả |
|---|---|
| < 1s | Serverless / scale-to-zero được |
| 1-10s | Cần warm pool, readiness probe phải chờ đủ lâu |
| **> 10s** | **Không scale-to-zero được.** Rolling deploy phải chờ instance mới ready rồi mới tắt cũ |

Đo nó ở `03_load_from_registry.py`, đừng đoán.

### 5. Streaming: đổi TTFT lấy TTLT

Streaming **không làm hệ thống nhanh hơn một mili giây nào**. Nó đổi
*time to first token* lấy *time to last token* (thường chậm hơn chút vì
overhead giao thức).

Nhưng user cảm nhận TTFT. 800ms thấy phản hồi ≫ 6 giây màn hình trắng.

Giao thức SSE — bốn chi tiết dễ sai:

```
event: sources
data: {"sources": [...]}          ← gửi TRƯỚC token, ngay sau retrieval

event: token
data: {"text": "Deep"}

event: done
data: {"ttft_ms": 812, "total_ms": 5230}    ← BẮT BUỘC phải có
```

1. **Hai dấu `\n\n`** kết thúc mỗi sự kiện. Thiếu một cái → client treo.
2. **Gửi `sources` trước** — UI có cái để hiện trong lúc chữ đang chảy.
3. **Phải có `done`** — không có nó, client không phân biệt được "xong" với
   "mất kết nối".
4. **Lỗi giữa chừng không đổi được status code.** Đã gửi 200 OK rồi. Lỗi phải
   đi qua sự kiện `error` trong luồng.

Thêm hai thứ thực chiến: header `X-Accel-Buffering: no` (nếu không, nginx sẽ
buffer hết và streaming coi như không có — **triệu chứng chỉ xuất hiện ở
production**), và `curl -N` khi test thủ công.

### 6. Phân loại mã lỗi cho đúng

| Mã | Nghĩa | Client nên làm gì |
|---|---|---|
| 422 | Client gửi sai schema | Sửa request, **đừng retry** |
| 429 | Quá nhiều request | Retry sau `Retry-After` |
| **503** | **Tạm thời chưa phục vụ được** (model đang load) | **Retry sau vài giây** |
| 500 | Lỗi nội bộ ngoài dự kiến | Retry một lần rồi báo lỗi |
| 504 | Upstream (LLM) timeout | Retry với backoff |

Trả 500 cho "model đang load" khiến client bỏ cuộc oan. Trả 503 cho lỗi
vĩnh viễn gây bão retry. Đây là chi tiết nhỏ mà người phỏng vấn có kinh
nghiệm luôn để ý.

### 7. Dependency injection — vì sao đáng công

```python
def get_pipeline(state: AppState = Depends(get_state)) -> RagPipeline: ...

@app.post("/query")
def query(req: QueryRequest, pipeline = Depends(get_pipeline)): ...

# Trong test:
app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
```

Bộ test chạy trong **dưới 1 giây** thay vì 30 giây load model thật. Với biến
toàn cục bạn phải monkeypatch, và các test bắt đầu ảnh hưởng lẫn nhau.

`06_fastapi_full.py --selftest` chứng minh điều này chạy được.

---

## Thứ tự chạy practice

| # | File | Học gì | Tốn API? |
|---|---|---|---|
| — | `rag_pipeline.py` | Pipeline RAG tối giản dùng chung — không phải bài học | — |
| 01 | `01_pyfunc_wrapper.py` | `load_context` vs `__init__`, signature, params | ~3 call |
| 02 | `02_log_and_register.py` | Log model, `code_paths`, đăng ký, gắn alias | **0** |
| 03 | `03_load_from_registry.py` | Load theo alias, cold-start, signature enforcement | ~1 call |
| 04 | `04_fastapi_minimal.py` | `/health` vs `/ready`, Pydantic, lifespan | ~2 call |
| 05 | `05_fastapi_streaming.py` | SSE, thứ tự sự kiện, TTFT vs total | ~2 call |
| 06 | `06_fastapi_full.py` | DI, request ID, exception handler, registry fallback | **0** (selftest) |

```powershell
python practice/04_serving/01_pyfunc_wrapper.py
python practice/04_serving/02_log_and_register.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
python practice/04_serving/03_load_from_registry.py
python practice/04_serving/04_fastapi_minimal.py --selftest
python practice/04_serving/05_fastapi_streaming.py --selftest
python practice/04_serving/06_fastapi_full.py --selftest

# Chạy server thật:
python practice/04_serving/06_fastapi_full.py --from-registry
# → http://127.0.0.1:8002/docs
```

Mọi file đều có `--selftest` chạy được **không cần curl và không cần terminal
thứ hai** — dùng `fastapi.testclient`.

---

## Ghép vào hệ thống

### Cấu trúc cần tạo

```
src/api/
  __init__.py
  main.py            ← create_app() factory + lifespan + middleware
  deps.py            ← get_state, get_pipeline, get_settings
  schemas.py         ← Pydantic request/response
  errors.py          ← exception handlers
  routes/
    health.py        ← /health, /ready
    query.py         ← /query, /query/stream
    ingest.py        ← /documents (Phase 6 mới đầy đủ)

src/serving/
  pyfunc.py          ← RagModel(mlflow.pyfunc.PythonModel)
  registry.py        ← load_from_registry, promote, resolve_alias
```

### Signature nên có

```python
# src/api/main.py
def create_app(settings: Settings | None = None) -> FastAPI: ...

# src/api/deps.py
def get_settings() -> Settings: ...
def get_state(request: Request) -> AppState: ...
def get_pipeline(state: AppState = Depends(get_state)) -> RagPipeline: ...

# src/serving/registry.py
def resolve_alias(name: str, alias: str = "champion") -> ModelInfo: ...
def load_config_from_registry(name: str, alias: str = "champion") -> RagConfig: ...
def promote(name: str, version: str, alias: str = "champion") -> None: ...
def promote_if_better(name: str, version: str, metric: str = "ndcg") -> bool: ...

# src/serving/pyfunc.py
class RagModel(mlflow.pyfunc.PythonModel): ...
def log_rag_model(config: RagConfig, registered_name: str) -> str: ...
```

### Nguyên tắc thiết kế phải giữ

1. **`create_app()` là factory, không tạo app ở module level.** Test cần dựng
   nhiều app độc lập; import không được có side effect.
2. **Load model ở luồng nền trong `lifespan`.** Server bind port ngay, `/health`
   trả lời được trong lúc model còn đang load — đúng ngữ nghĩa liveness.
3. **Load thất bại KHÔNG được ném exception ra khỏi lifespan.** Ghi lỗi vào
   state, `/ready` trả 503 kèm nguyên nhân. Ném ra → container chết → orchestrator
   restart → chết lại. Vòng lặp.
4. **Registry chết không được làm chết API.** Fallback về cấu hình local, log
   cảnh báo to, đặt `model_version = "local-fallback"` để nhìn là biết.
5. **Lấy CẤU HÌNH từ registry rồi build lại**, đừng `pyfunc.load_model()` trong
   API. Vì pipeline cần instrument được (Phase 5) và index thật nằm ở Qdrant,
   không nằm trong artifact.
6. **Request ID trên mọi request, mọi log line, và trong response body.** Đây
   chính là `trace_id` mà Phase 5 dùng để nối log với trace.
7. **Log đầy đủ ở server, gửi tối giản cho client.** Stack trace lộ cấu trúc
   nội bộ.

### Sửa code cũ

| File | Việc cần làm |
|---|---|
| `src/app.py` (Streamlit) | Gọi API qua HTTP thay vì import trực tiếp `rag_chatbot`. Streamlit thành **client**, không còn là hệ thống. |
| `src/app_cli.py` | Tương tự — hoặc giữ nguyên làm công cụ debug offline. |
| `src/rag/rag_chatbot.py` | Bỏ singleton. API tự quản vòng đời pipeline qua `app.state`. |

Chuyển Streamlit sang gọi API là bước quan trọng: nó chứng minh kiến trúc của
bạn có **biên giới rõ ràng**, không phải một khối code dính chặt.

---

## Bẫy thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Artifact nặng vài trăm MB | Gán index/model vào `__init__` | Chuyển sang `load_context` |
| `ModuleNotFoundError: rag_pipeline` khi load ở máy khác | Thiếu `code_paths` | `code_paths=[".../rag_pipeline.py"]` |
| Container restart vô hạn | Liveness probe kiểm tra model | Tách `/health` khỏi `/ready` |
| 500 hàng loạt mỗi lần deploy | Không có readiness probe | Thêm `/ready`, cấu hình LB đọc nó |
| Streaming không chảy khi test bằng curl | Thiếu `-N` | `curl -N` |
| Streaming không chảy **chỉ ở production** | nginx buffer | Header `X-Accel-Buffering: no` |
| Client treo, chờ mãi | Thiếu `\n\n` cuối sự kiện SSE | `f"event: {e}\ndata: {json}\n\n"` |
| Client không biết khi nào xong | Thiếu sự kiện `done` | Luôn gửi `done` ở cuối |
| Đổi alias nhưng API vẫn hành vi cũ | Model đã nằm trong RAM | Restart, hoặc thêm `/admin/reload` |
| Test chạy 30 giây mỗi lần | Không dùng DI | `app.dependency_overrides` |
| MLflow log_model cảnh báo về `artifact_path` | API cũ | MLflow 3.x dùng `name=` thay `artifact_path=` |

---

## Definition of Done

- [ ] `python_model.pkl` trong artifact **< 100 KB** (kiểm tra bằng file 02 demo 3).
- [ ] Registry có ≥ 2 version, `@champion` và `@challenger` gắn đúng chỗ.
- [ ] `POST /query` trả `answer`, `sources[]`, `latency_ms`, `model_version`, `request_id`.
- [ ] Streaming chạy — nhìn thấy token chảy, và ttft < total.
- [ ] `/health` trả 200 trong khi `/ready` trả 503 lúc đang load. **Tự tay
      quan sát được cửa sổ thời gian này.**
- [ ] Đổi alias `@champion` → restart API → `/ready` báo `model_version` mới và
      hành vi đổi. **Không sửa dòng code nào.**
- [ ] Bộ test dùng `dependency_overrides` chạy dưới 2 giây.
- [ ] Streamlit gọi qua API, không import trực tiếp.
- [ ] Biết cold-start của mình là bao nhiêu giây.

**Câu cho CV** (điền số thật):

> Đóng gói RAG pipeline thành MLflow pyfunc model (signature + pinned
> requirements + code_paths), serve qua FastAPI với SSE streaming — TTFT __ms
> so với __ms end-to-end. Deploy bằng alias registry: đổi `@champion` là xong,
> không đổi code, rollback trong vài giây. Tách liveness/readiness đúng ngữ
> nghĩa cho cold-start __s.

---

## Câu hỏi phỏng vấn

**1. Vì sao `/health` và `/ready` phải tách?**
Vì hệ quả của FAIL khác nhau hoàn toàn. Liveness FAIL thì orchestrator giết
container; readiness FAIL thì load balancer chỉ ngừng gửi traffic. RAG có
cold-start 10-30 giây, nên nếu liveness kiểm tra model thì container sẽ bị
giết ngay trong lúc đang load — và lặp lại vĩnh viễn.

**2. Deploy model mới thế nào?**
Log version mới vào registry, chạy eval, nếu tốt hơn thì đổi alias `@champion`
sang version đó rồi restart API. Không build image, không merge code. Rollback
là gắn alias về version cũ, vài giây.

**3. Cái gì được đóng gói trong pyfunc model?**
Chỉ cấu hình, signature, requirements đã pin, và code cần để import lại. Index
và embedding model **không** — chúng được dựng trong `load_context` lúc serve.
Nếu chúng lọt vào `__init__` thì bị pickle và artifact phình lên hàng trăm MB
với một index đã lỗi thời.

**4. Streaming làm hệ thống nhanh hơn à?**
Không. Tổng thời gian thường còn chậm hơn chút vì overhead giao thức. Nó đổi
time-to-first-token lấy time-to-last-token. User cảm nhận TTFT, nên UX tốt lên
nhiều dù không có mili giây nào được tiết kiệm.

**5. Lỗi xảy ra giữa chừng khi đang stream thì sao?**
Không đổi được status code nữa vì 200 OK đã gửi. Lỗi phải đi qua một sự kiện
`error` trong luồng SSE, và client phải xử lý được. Đây là khác biệt cơ bản
so với API thường và cũng là chỗ hay bị bỏ sót.

**6. Vì sao dùng dependency injection thay vì biến toàn cục?**
Chủ yếu vì test. Với `dependency_overrides` tôi thay pipeline thật bằng fake
trong một dòng, bộ test chạy dưới một giây. Biến toàn cục thì phải monkeypatch
và các test bắt đầu ảnh hưởng lẫn nhau.

**7. MLflow registry chết thì API của bạn ra sao?**
Vẫn khởi động được. Nó log cảnh báo rồi chạy bằng cấu hình local, và đặt
`model_version = "local-fallback"` để nhìn `/ready` là biết ngay đang ở chế
độ suy giảm. Registry là nguồn chân lý về cấu hình, nhưng không được là điểm
chết đơn.

**8. Điều tra một câu trả lời sai của tuần trước thế nào?**
Lấy `request_id` từ user, grep log ra được câu hỏi, các chunk đã lấy, và
`model_version` lúc đó. Quan trọng là log **số version**, không phải alias —
alias có thể đã đổi và load `@champion` bây giờ sẽ ra model khác.

---

← [Phase 3 — A/B Testing](03_ab_testing.md) · [Phase 5 — Observability](05_observability.md) →
