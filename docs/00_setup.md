# Phase 0 — Nền móng & Trả nợ kỹ thuật

> **Thời gian**: ~2 ngày
> **Điều kiện tiên quyết**: không có
> **Kết quả**: môi trường sạch, embedding chạy đúng, provider layer tách bạch

---

## 1. Mục tiêu phase

Sau phase này bạn sẽ có:

- Môi trường Python 3.11 sạch, cài đủ, kiểm tra được bằng một lệnh.
- **Embedding chạy local** — sửa lỗi nghiêm trọng nhất của codebase hiện tại.
- Một `provider layer` (`src/core/`) cho phép đổi LLM/embedding bằng config,
  không sửa code. Đây là điều kiện **bắt buộc** để A/B test ở Phase 3.
- Hiểu tận gốc một lời gọi LLM và một lời gọi embedding thực sự là gì.

Nghe có vẻ nhàm. Nhưng bỏ qua phase này thì mọi con số ở Phase 1 và Phase 3
đều là số rác, và bạn sẽ mang số rác đó đi phỏng vấn.

---

## 2. Nền tảng lý thuyết

### 2.1. Vì sao OpenRouter không làm được embedding

Đây là lỗi thật đang nằm trong `src/rag/rag_chatbot.py`:

```python
Settings.embed_model = OpenAIEmbedding(
    api_base="https://openrouter.ai/api/v1",   # ❌
    api_key=self.api_key,                       # ❌ đây là OpenRouter key
    model=OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
)
```

**Vì sao sai**: OpenRouter là một **router cho chat completions**. Nó nhận
request theo format OpenAI rồi chuyển tiếp tới Anthropic, Google, Meta, DeepSeek...
Nhưng nó chỉ route **một** endpoint: `POST /chat/completions`.

Không có `POST /embeddings`. Vì:
- Embedding model không "chat" — chúng là encoder, không phải decoder.
- Mỗi provider có format embedding riêng, không có chuẩn chung để route.
- Kinh tế: embedding cực rẻ, không đáng để làm proxy.

**Hậu quả của lỗi này** — và đây là phần nguy hiểm:

| Kịch bản | Điều xảy ra |
|---|---|
| Tốt nhất | Throw 404 ngay → bạn biết mà sửa |
| Xấu | Request treo tới timeout, retry 2 lần, chậm 3× |
| Tệ nhất | Client fallback về `api.openai.com` với OpenRouter key → 401, nhưng lỗi bị nuốt ở tầng sâu và bạn tưởng vấn đề nằm ở chunking |

Bài `01_openrouter_hello.py` **chứng minh bằng thực nghiệm** điều này — bạn
tự gọi endpoint và xem status code, không phải tin lời tôi.

### 2.2. Bi-encoder vs Cross-encoder — hiểu từ bây giờ

Đây là kiến thức nền cho cả Phase 3.

```
BI-ENCODER (embedding thường)
  query    ──► [encoder] ──► vector_q  ┐
                                       ├──► cosine(q, d)
  document ──► [encoder] ──► vector_d  ┘

  • Encode document MỘT LẦN, lưu vào vector store.
  • Query đến: encode 1 lần, so với 1 triệu vector trong vài ms.
  • → Nhanh, scale được. Nhưng query và document không "nhìn thấy nhau"
    lúc encode → bỏ sót sắc thái.

CROSS-ENCODER (reranker)
  [query + document] ──► [encoder] ──► điểm liên quan

  • Đọc CẢ CẶP cùng lúc → attention chạy chéo giữa hai bên → chính xác hơn nhiều.
  • Nhưng: không precompute được. 1 triệu document = 1 triệu lần forward pass.
  • → Chỉ dùng để rerank top-50 mà bi-encoder đã lọc ra.
```

Kiến trúc chuẩn công nghiệp là **hai tầng**:

```
1 triệu docs ──[bi-encoder, 5ms]──► top 50 ──[cross-encoder, 80ms]──► top 5 ──► LLM
```

Phase 3 sẽ đo chính xác tầng thứ hai đáng giá bao nhiêu.

### 2.3. Vì sao embedding local là lựa chọn đúng cho project này

| Tiêu chí | Local (bge-small) | OpenAI API |
|---|---|---|
| Chi phí index 100k chunk | **$0** | ~$0.40 |
| Chi phí A/B test 6 cấu hình chunking | **$0** | ~$2.40 mỗi vòng |
| Chi phí re-index khi đổi embedding | **$0** | tính lại từ đầu |
| Độ trễ mỗi query | ~5ms | ~80ms (round-trip mạng) |
| Chạy offline | ✅ | ❌ |
| Chất lượng (MTEB) | tốt | tốt hơn ~3-5% |
| Cần cài torch (~500MB CPU) | ✅ phải cài | không |

Điểm quyết định là **dòng thứ hai**. Cả roadmap này xoay quanh việc đo đạc
và A/B test. Nếu mỗi lần đổi chunk_size lại tốn tiền, bạn sẽ tự giới hạn
số thí nghiệm — và đó chính là thứ làm project mất giá trị.

Đánh đổi 3-5% chất lượng để có thể chạy 24 thí nghiệm thay vì 3 là một
đánh đổi tốt. **Và bạn nói được câu đó trong buổi phỏng vấn** — đó mới là
điều quan trọng.

### 2.4. Provider layer — vì sao phải tách

Code hiện tại nhúng cứng việc tạo model vào trong `RAGChatBot.__init__`:

```python
def _setup_embedding(self):
    Settings.embed_model = OpenAIEmbedding(...)   # cứng
```

Muốn A/B test 4 embedding model, bạn phải sửa class này 4 lần. Không làm được.

Kiến trúc đúng:

```
        config (dict / env / MLflow params)
                     │
                     ▼
        ┌────────────────────────┐
        │  src/core/providers.py │   ← factory thuần, không state
        │  get_llm(name)         │
        │  get_embedding(name)   │
        └────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
   RAG pipeline            Eval harness
   (production)            (Phase 1, 3)
```

Nguyên tắc: **mọi thứ có thể thay đổi trong thí nghiệm phải đi vào qua tham số**,
không được nằm cứng trong constructor.

### 2.5. Bẫy singleton

```python
_chatbot_instance = None

def get_chatbot(data_dir=None, chunk_size=200, ...):
    global _chatbot_instance
    if _chatbot_instance is None:          # ⚠️
        _chatbot_instance = RAGChatBot(chunk_size=chunk_size, ...)
    return _chatbot_instance                # tham số bị BỎ QUA từ lần 2 trở đi
```

Trong `src/app.py` (Streamlit) thì vô hại. Nhưng ở Phase 3:

```python
for size in [128, 256, 512]:
    bot = get_chatbot(chunk_size=size)   # lần 2, 3 nhận lại bot chunk_size=128
    results.append(evaluate(bot))         # → 3 kết quả GIỐNG HỆT NHAU
```

Bạn sẽ có một bảng A/B đẹp đẽ với ba dòng số liệu giống nhau và không hiểu vì sao.
Đây là loại bug **không crash, chỉ nói dối**. Nguy hiểm nhất trong ML.

Cách sửa: bỏ singleton, để caller tự quyết định cache (Streamlit đã có
`@st.cache_resource` rồi).

---

## 3. Thứ tự chạy practice

| # | File | Học gì | Lệnh |
|---|---|---|---|
| 0 | `00_env_check.py` | Preflight check: fail sớm, fail rõ | `python practice/00_setup/00_env_check.py` |
| 1 | `01_openrouter_hello.py` | HTTP trần, messages, usage, temperature, chứng minh không có /embeddings | `python practice/00_setup/01_openrouter_hello.py` |
| 2 | `02_local_embedding.py` | Load model local, vector, cosine, batch, BGE prefix | `python practice/00_setup/02_local_embedding.py` |
| 3 | `03_embedding_compare.py` | Mini A/B test 4 model — bản thu nhỏ của Phase 3 | `python practice/00_setup/03_embedding_compare.py` |

Chi phí: bài 1 tốn ~$0.001. Bài 2, 3 hoàn toàn miễn phí.

### Cài đặt môi trường

```powershell
# 1. Tạo venv Python 3.11 (thư mục venv_311 hiện tại đang chứa 3.14 — bỏ đi)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 2. Cài torch CPU-only TRƯỚC. Nếu không, pip kéo bản CUDA ~2.5GB vô ích.
pip install --index-url https://download.pytorch.org/whl/cpu torch

# 3. Cài phần còn lại
pip install -r requirements/base.txt

# 4. Tạo .env
copy .env.example .env
#    rồi mở ra điền OPENROUTER_API_KEY

# 5. Kiểm tra
python practice/00_setup/00_env_check.py
```

---

## 4. Ghép vào hệ thống

Sau khi chạy hết practice, tự viết code thật trong `src/core/`.

### Cấu trúc cần tạo

```
src/core/
  __init__.py
  settings.py      # cấu hình tập trung, đọc từ env
  providers.py     # factory tạo LLM / embedding
  errors.py        # exception riêng của project
```

### `src/core/settings.py`

Dùng `pydantic-settings` — nó validate kiểu và đọc `.env` tự động.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str
    gen_model: str = "inclusionai/ling-3.0-flash"
    judge_model: str = "google/gemini-2.5-flash"
    embedding_model: str = "bge-small"

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5

    qdrant_url: str = "http://localhost:6333"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Gợi ý: thêm @field_validator để chặn gen_model == judge_model

settings = Settings()      # import chỗ này ở mọi nơi khác
```

**Vì sao pydantic-settings chứ không phải `os.getenv`**: nó fail lúc khởi động
với thông báo rõ ràng nếu thiếu biến, thay vì fail lúc 3 giờ sáng khi request
thứ 400 đi vào nhánh code hiếm.

### `src/core/providers.py`

Signature cần có (bạn tự viết phần thân):

```python
EMBEDDING_REGISTRY: dict[str, EmbeddingSpec]
    # tên ngắn -> (repo HF, số chiều, query_instruction)

def get_llm(model: str | None = None, **overrides) -> LLM: ...
def get_embedding(name: str | None = None, **overrides) -> BaseEmbedding: ...
def get_embedding_dim(name: str | None = None) -> int: ...
def configure_llama_settings(embedding=None, model=None) -> None: ...
```

Yêu cầu:
- Cache embedding model theo tên (load lại tốn 2 giây mỗi lần).
- `get_embedding` phải set `query_instruction` đúng cho model BGE.
- Không hàm nào được đọc env trực tiếp — chỉ đọc qua `settings`.

### Sửa code cũ

| File | Việc phải làm |
|---|---|
| `src/rag/rag_chatbot.py` | Xoá `_setup_embedding` cũ, gọi `configure_llama_settings()`. Xoá `_chatbot_instance` và `get_chatbot()`. Nhận `embedding_name` làm tham số constructor. |
| `src/rag/main.py` | Bỏ đường dẫn tuyệt đối `D:\Projects\...`, dùng `DATA_DIR`. Bỏ `OpenAIEmbedding`. |
| `src/rag/config.py` | Xoá — thay bằng `src/core/settings.py`. |
| `src/config.py` | Xoá — `POROJECT_ROOT = './'` vừa sai chính tả vừa sai giá trị. |
| `src/app.py` | Đổi `get_chatbot(...)` thành `RAGChatBot(...)`, giữ `@st.cache_resource`. |
| `src/app_cli.py` | Như trên. |

### Kiểm chứng đã sửa xong

```powershell
# Không được có kết quả nào:
findstr /s /i "openrouter.ai/api/v1" src\*.py | findstr /i "embed"

# Cũng không được có:
findstr /s /i "_chatbot_instance" src\*.py
findstr /s /i "D:\\Projects" src\*.py
```

---

## 5. Bẫy thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `pip install torch` tải 2.5GB rồi lỗi | pip lấy bản CUDA | `pip install --index-url https://download.pytorch.org/whl/cpu torch` |
| `OSError: [WinError 1314]` khi tải model HF | Windows chặn symlink | Đặt env `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, hoặc bật Developer Mode |
| Model tải lại mỗi lần chạy | `HF_HOME` trỏ vào thư mục tạm | Đặt `HF_HOME` cố định trong `.env` |
| `401 Unauthorized` từ OpenRouter | Key sai / hết hạn / hết credit | Kiểm tra bằng `GET /api/v1/auth/key` (bài tập 2 của bài 1) |
| Embedding rất chậm (>1s/câu) | Đang encode từng câu, không batch | Dùng `get_text_embedding_batch()` |
| `ImportError: sentence_transformers` | Chưa cài, hoặc torch lỗi | Cài torch trước, rồi `pip install sentence-transformers` |
| Retrieval kém dù model tốt | Thiếu `query_instruction` với BGE | Xem demo 5 của bài 2 |
| Chunk dài hơn 512 token bị cắt âm thầm | `max_seq_length` của model | Đặt `chunk_size` nhỏ hơn giới hạn model |
| A/B test cho 3 kết quả giống hệt | Singleton `_chatbot_instance` | Xoá singleton |

---

## 6. Definition of Done

- [ ] `.venv` dùng **Python 3.11**, `python practice/00_setup/00_env_check.py` báo xanh hết
- [ ] Cả 4 file practice chạy xong không lỗi
- [ ] Chạy `src/app_cli.py` và **không có lời gọi mạng nào cho embedding**
      (kiểm tra: ngắt mạng sau khi model đã tải → index vẫn build được)
- [ ] Đổi embedding model chỉ bằng sửa `EMBEDDING_MODEL` trong `.env`, không sửa code
- [ ] `_chatbot_instance` đã bị xoá khỏi codebase
- [ ] Không còn đường dẫn tuyệt đối nào trong `src/`
- [ ] `src/core/settings.py` fail rõ ràng khi thiếu `OPENROUTER_API_KEY`
- [ ] Ghi lại **baseline** vào bảng ở `ROADMAP.md` mục 6: thời gian build index,
      số chunk, embedding đang dùng

---

## 7. Câu hỏi phỏng vấn có thể gặp

**1. "Vì sao bạn chọn embedding local thay vì OpenAI API?"**
> Vì kiến trúc của project xoay quanh A/B testing. Tôi chạy 24 cấu hình
> chunking × embedding, mỗi lần đổi là phải re-index toàn bộ corpus. Với API
> tính tiền theo token thì chi phí thí nghiệm trở thành rào cản và tôi sẽ tự
> giới hạn số thí nghiệm. Local cho tôi chạy không giới hạn với giá 3-5% chất
> lượng — tôi đo được con số đó ở Phase 3, không phải đoán. Nếu lên production
> thật với SLA cao thì tôi sẽ cân nhắc lại, và tôi đã có sẵn số liệu để quyết định.

**2. "Khác nhau giữa bi-encoder và cross-encoder?"**
> Bi-encoder encode query và document độc lập nên precompute được — đó là cách
> vector database hoạt động, tìm trong 1 triệu document mất vài ms. Cross-encoder
> đọc cả cặp cùng lúc nên attention chạy chéo, chính xác hơn đáng kể, nhưng không
> precompute được. Kiến trúc chuẩn là hai tầng: bi-encoder lọc top-50, cross-encoder
> rerank xuống top-5. Trong project tôi đo được reranker nâng NDCG@5 nhưng thêm
> ~180ms p95 — và tôi có biểu đồ Pareto để giải thích lựa chọn.

**3. "Cosine similarity đo cái gì? Nó có nhược điểm gì?"**
> Nó đo góc giữa hai vector, tức độ giống về chủ đề. Nhược điểm lớn nhất là nó
> không hiểu logic: "bạn nên train model" và "bạn không bao giờ nên train model"
> có cosine rất cao dù nghĩa ngược nhau. Tôi có demo đo được điều này. Đó là
> giới hạn cơ bản của dense retrieval và là lý do cần reranker hoặc hybrid search.

**4. "Điểm cosine của bạn là 0.85, thế là tốt hay xấu?"**
> Câu này không trả lời được nếu không biết model nào. Thang điểm cosine không
> so sánh được giữa các embedding model khác nhau — chỉ thứ hạng mới so sánh được.
> Đây là lý do tôi đo hit rate và NDCG chứ không đặt ngưỡng cosine cứng. Nếu ai
> đó đặt threshold 0.7 rồi đổi embedding model mà quên chỉnh, hệ thống sẽ hỏng âm thầm.

**5. "Vì sao dùng Python 3.11 mà không phải bản mới nhất?"**
> Vì toàn bộ ML stack — torch, sentence-transformers, các thư viện OTel
> instrumentation — có wheel dựng sẵn và được test kỹ trên 3.11. Bản mới hơn
> thường phải build từ source hoặc chưa hỗ trợ. Trong dự án production tôi ưu tiên
> phiên bản có hệ sinh thái ổn định hơn là phiên bản mới nhất.

**6. "Kể một bug bạn tìm ra trong project này."**
> Codebase gọi embedding qua base URL của OpenRouter, nhưng OpenRouter chỉ proxy
> chat completions, không có endpoint embeddings. Bug này không crash rõ ràng —
> nó thất bại ở tầng sâu trong framework. Tôi phát hiện bằng cách gọi thẳng
> endpoint bằng httpx để xem status code thật thay vì tin vào abstraction.
> Bài học: khi debug hệ thống nhiều tầng, hãy xuống tầng thấp nhất bạn kiểm soát được.

**7. "Vì sao bỏ singleton?"**
> Vì `get_chatbot()` bỏ qua tham số từ lần gọi thứ hai trở đi. Trong Streamlit
> thì vô hại, nhưng khi tôi chạy A/B test 3 chunk_size khác nhau, cả 3 đều nhận
> lại instance đầu tiên và cho ra kết quả giống hệt nhau. Đây là loại bug không
> crash mà chỉ nói dối — nguy hiểm nhất trong ML, vì bạn có một bảng số liệu trông
> hoàn toàn hợp lý nhưng sai. Tôi thay bằng factory thuần và để caller tự quản lý cache.

---

## 8. Tiếp theo

→ [`docs/01_evaluation.md`](01_evaluation.md) — Phase 1: Evaluation harness.

Trước khi sang, điền baseline vào bảng mục 6 của `ROADMAP.md`. Không có
baseline thì không chứng minh được cải thiện.
