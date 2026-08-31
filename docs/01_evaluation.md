# Phase 1 — Evaluation Harness

> Phase quan trọng nhất của cả project. Mọi con số bạn đưa vào CV đều sinh ra từ đây.
> Nếu chỉ có thời gian làm 2 phase, làm Phase 0 và phase này.

---

## Mục tiêu

Kết thúc phase này bạn có:

1. Một **golden dataset** 50+ câu hỏi, sinh từ `data/`, đã qua kiểm định tự động.
2. **5 retrieval metric** tự cài bằng numpy: Hit Rate, MRR, Precision, Recall, NDCG.
3. **2 generation metric** dùng LLM-as-judge: Faithfulness, Answer Relevancy.
4. Một **judge có schema JSON**, parse an toàn, tự retry khi model trả rác.
5. Bằng chứng **judge của bạn đáng tin** — đo variance qua nhiều lần chạy.
6. Toàn bộ log vào **MLflow**, so được run này với run kia.

Không có phase này thì Phase 2 (prompt optimization) và Phase 3 (A/B testing)
đều vô nghĩa: bạn sẽ đổi cấu hình mà không biết nó tốt lên hay xấu đi.

---

## Nền tảng lý thuyết

### 1. RAG có hai tầng, phải đo tách riêng

```
   câu hỏi
      │
      ▼
┌─────────────────┐
│   RETRIEVAL     │  ← đo bằng Hit Rate / MRR / NDCG
│  (tìm chunk)    │     KHÔNG cần LLM. Xác định. Rẻ. Chạy 1000 lần / phút.
└────────┬────────┘
         │  top-k chunk
         ▼
┌─────────────────┐
│   GENERATION    │  ← đo bằng Faithfulness / Answer Relevancy
│  (LLM viết câu) │     CẦN LLM làm giám khảo. Ngẫu nhiên. Tốn tiền. Chậm.
└────────┬────────┘
         │
      câu trả lời
```

**Vì sao phải tách?** Vì khi điểm tổng tụt, bạn cần biết tụt ở đâu.

| Triệu chứng | Retrieval | Generation | Nguyên nhân thật |
|---|---|---|---|
| Trả lời sai bét | thấp | — | Không tìm ra đoạn đúng → sửa chunker / embedding |
| Trả lời trôi chảy nhưng bịa | **cao** | thấp | Tìm đúng nhưng LLM bịa → sửa prompt |
| Trả lời đúng nhưng lạc đề | cao | faithful cao, relevancy thấp | LLM trả lời câu khác → sửa prompt |
| Đúng hết | cao | cao | Xong. Đi tối ưu latency. |

Một điểm số duy nhất (kiểu "accuracy 78%") không phân biệt được 4 dòng trên.
Đây cũng là câu hỏi phỏng vấn hay gặp nhất của phase này.

### 2. Vì sao tự viết judge thay vì dùng RAGAS / DeepEval

Bạn **được phép** dùng thư viện trong production. Nhưng ở đây tự viết vì:

- Người phỏng vấn hỏi "faithfulness tính thế nào?" — nếu bạn chỉ gọi `ragas.evaluate()`
  thì không trả lời được, và đó là dấu hiệu rõ nhất của người mới.
- Thư viện eval thay đổi API liên tục; hiểu cơ chế thì đổi thư viện nào cũng dùng được.
- Judge tự viết cache được, kiểm soát chi phí được, ép schema được.

Faithfulness thật ra chỉ là 2 lần gọi LLM:

```
answer  ──[LLM]──►  danh sách claim nguyên tử
                          │
              với mỗi claim: context có chống lưng không? (yes/no)
                          │
                    faithfulness = #yes / #tổng
```

Answer Relevancy thì ngược chiều — đây là mẹo hay:

```
answer  ──[LLM]──►  sinh ngược N câu hỏi mà answer này trả lời được
                          │
        cosine(câu hỏi gốc, N câu hỏi sinh ngược) → trung bình
```

Nếu câu trả lời lạc đề, câu hỏi sinh ngược sẽ khác xa câu hỏi gốc → điểm thấp.
Không cần ground truth. Đây là lý do metric này chạy được cả trên production traffic.

### 3. Ba thiên kiến của LLM-as-judge

| Thiên kiến | Biểu hiện | Cách giảm |
|---|---|---|
| **Position bias** | Đưa A trước B thì A hay thắng | Chấm từng câu độc lập, hoặc hoán vị rồi lấy trung bình |
| **Verbosity bias** | Câu dài hơn được điểm cao hơn dù không đúng hơn | Rubric ghi rõ "độ dài không tính điểm" |
| **Self-preference bias** | Model chấm chính output của nó cao hơn | **Judge model PHẢI khác generator model** |

Trong `.env` của bạn:

```
GEN_MODEL=inclusionai/ling-3.0-flash     # người trả lời
JUDGE_MODEL=google/gemini-2.5-flash      # giám khảo — KHÁC nhà cung cấp
```

`practice/00_setup/00_env_check.py` sẽ cảnh báo nếu hai biến này bằng nhau.
File `01_llm_judge_basics.py` có demo đo thẳng self-preference bias.

### 4. Judge cũng cần được kiểm định

Judge là một cái cân. Cân cũng sai số. Trước khi tin số nó đưa ra, đo:

- **Variance**: chấm cùng một câu 8 lần, độ lệch chuẩn bao nhiêu?
  Nếu σ = 1.5 trên thang 5 thì chênh 0.3 điểm giữa hai cấu hình A/B là **nhiễu**.
- **Ảnh hưởng của temperature**: `temperature=0` giảm variance rất nhiều nhưng
  không về 0 (model vẫn không tất định hoàn toàn).
- **Độ nhạy với prompt**: đổi một câu trong rubric mà điểm nhảy → rubric chưa chặt.

File `07_judge_reliability.py` làm đúng ba việc trên và in ra
"khác biệt tối thiểu có ý nghĩa" — con số bạn cần để đọc bảng A/B ở Phase 3.

### 5. Golden dataset — chỗ dễ tự lừa mình nhất

Sinh câu hỏi bằng LLM thì nhanh, nhưng dễ ra bộ dữ liệu vô dụng. Bốn lỗi kinh điển,
`05_golden_dataset.py` kiểm tra tự động cả bốn:

| Lỗi | Ví dụ | Vì sao chết |
|---|---|---|
| **Self-referential** | "Theo đoạn văn trên, X là gì?" | Câu hỏi thật của user không bao giờ có "đoạn văn trên" |
| **Leak từ vựng** | Câu hỏi trùng >75% từ với chunk | Retrieval nào cũng tìm ra → metric luôn ~1.0, vô nghĩa |
| **Trùng lặp** | Hai câu hỏi cosine > 0.93 | Cùng một câu đếm 2 lần → trọng số lệch |
| **Không truy hồi được** | Chunk gán nhãn không nằm trong top-5 | Nhãn sai, không phải retrieval kém |

Sau khi lọc, 100 câu sinh ra thường còn 50-60 câu dùng được. Đó là bình thường.

**Cỡ mẫu tối thiểu.** Với 20 câu, sai số ±11%. Với 50 câu, ±7%. Với 200 câu, ±3.5%.
Nếu hai cấu hình chênh 4% mà bạn chỉ có 30 câu thì bạn không kết luận được gì —
Phase 3 dùng bootstrap CI để nói chính xác điều này.

---

## Thứ tự chạy practice

Chạy đúng thứ tự — file sau dùng khái niệm của file trước.

| # | File | Học gì | Tốn API? |
|---|---|---|---|
| 01 | `01_llm_judge_basics.py` | Judge chỉ là một prompt. Rubric vs prompt mơ hồ. Self-preference bias. | ~20 call |
| 02 | `02_structured_judge.py` | Ép JSON schema, parse chịu lỗi, retry với repair prompt | ~15 call |
| 03 | `03_retrieval_metrics.py` | 5 metric bằng numpy thuần, có ví dụ tính tay | **0** — offline |
| 04 | `04_generation_metrics.py` | Faithfulness (tách claim → verify) + Answer Relevancy (sinh ngược) | ~30 call |
| 05 | `05_golden_dataset.py` | Sinh + validate golden set, ghi `eval_data/golden_set.json` | ~40 call |
| 06 | `06_mlflow_eval_run.py` | Ghép tất cả thành 1 pipeline, log vào MLflow | ~60 call |
| 07 | `07_judge_reliability.py` | Đo variance của judge, tính ngưỡng khác biệt có ý nghĩa | ~50 call |

```powershell
python practice/01_evaluation/03_retrieval_metrics.py   # chạy file này TRƯỚC — miễn phí
python practice/01_evaluation/01_llm_judge_basics.py
python practice/01_evaluation/02_structured_judge.py
python practice/01_evaluation/04_generation_metrics.py
python practice/01_evaluation/05_golden_dataset.py
python practice/01_evaluation/06_mlflow_eval_run.py
mlflow ui --backend-store-uri sqlite:///mlflow.db       # mở http://localhost:5000
python practice/01_evaluation/07_judge_reliability.py
```

Toàn bộ judge response được cache vào `.cache/` qua `disk_cache()`. Chạy lại lần 2
gần như miễn phí. Muốn bỏ cache: `$env:NO_CACHE=1`.

---

## Ghép vào hệ thống

Đây là phần **bạn tự viết**. Practice dạy khái niệm; `src/` là nơi bạn dựng thật.

### Cấu trúc cần tạo

```
src/eval/
  __init__.py
  metrics/
    retrieval.py      ← bê từ practice 03, không sửa gì nhiều
    generation.py     ← faithfulness + answer_relevancy từ practice 04
  judge.py            ← judge có schema từ practice 02
  dataset.py          ← load/validate golden set từ practice 05
  runner.py           ← pipeline + log MLflow từ practice 06
```

### Signature nên có

```python
# src/eval/metrics/retrieval.py
def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float: ...
def mrr_at_k(...)        -> float: ...
def ndcg_at_k(...)       -> float: ...
def precision_at_k(...)  -> float: ...
def recall_at_k(...)     -> float: ...

# src/eval/judge.py
@dataclass(frozen=True)
class JudgeResult:
    score: float
    reasoning: str
    raw: dict
    n_retries: int

class Judge:
    def __init__(self, model: str | None = None, schema: dict | None = None,
                 max_retries: int = 2, cache: bool = True): ...
    def score(self, question: str, answer: str, context: str) -> JudgeResult: ...

# src/eval/dataset.py
@dataclass(frozen=True)
class GoldenCase:
    question: str
    ground_truth: str
    relevant_chunk_ids: list[str]
    source: str

def load_golden_set(path: Path) -> list[GoldenCase]: ...
def validate(cases: list[GoldenCase], corpus: list[dict]) -> ValidationReport: ...

# src/eval/runner.py
@dataclass
class EvalConfig:
    chunker: str; chunk_size: int; embedding: str
    top_k: int; gen_model: str; judge_model: str; prompt_version: str

def run_eval(config: EvalConfig, cases: list[GoldenCase],
             log_to_mlflow: bool = True) -> EvalReport: ...
```

`EvalConfig` là **hợp đồng với Phase 3**. Mọi thứ Phase 3 muốn A/B đều phải là một
trường trong đây. Thiết kế nó rộng ngay từ giờ, đừng để Phase 3 phải sửa lại runner.

### Nguyên tắc thiết kế phải giữ

1. **`run_eval` nhận config, không đọc biến toàn cục.** Nếu nó đọc `Settings` toàn cục
   thì Phase 3 chạy 6 cấu hình trong 1 process sẽ ra 6 kết quả giống hệt nhau — đúng
   cái bẫy singleton đã nói ở `docs/00_setup.md`.
2. **Tách "tính toán" khỏi "tác dụng phụ".** `run_eval` trả về `EvalReport`;
   `log_to_mlflow(report)` là hàm riêng. Như vậy test được `run_eval` không cần MLflow.
3. **Lưu per-query, không chỉ trung bình.** Trung bình giấu mất câu hỏi thất bại.
   Log `per_query.json` làm MLflow artifact — đây là thứ dùng để debug và cũng là
   thứ gây ấn tượng khi demo.
4. **Cache judge response theo hash(prompt + model).** Chạy lại eval khi chỉ sửa
   retrieval thì không nên trả tiền judge lần nữa.

### Sửa code cũ

| File | Việc cần làm |
|---|---|
| `src/rag/rag_chatbot.py` | Tách hàm `retrieve(question, k) -> list[Node]` ra khỏi hàm trả lời. Eval cần chạm được tầng retrieval riêng. |
| `src/rag/rag_chatbot.py` | Bỏ `_chatbot_instance` singleton, hoặc thêm tham số `fresh=True`. |
| `src/rag/chunking.py` | Giữ nguyên — `ChunkerFactory` đã đúng hình dạng Phase 3 cần. |

---

## Bẫy thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Judge trả về text kèm ```` ```json ```` | Model nào cũng thích bọc fence | `extract_json()` ở practice 02 đã xử lý |
| `JSONDecodeError` ngẫu nhiên 5% số lần | Trailing comma, comment `//`, cắt giữa chừng | Retry với repair prompt, tối đa 2 lần |
| Hit Rate = 1.0 trên mọi cấu hình | Golden set bị leak từ vựng | Chạy validator ở practice 05, bỏ câu overlap > 0.75 |
| Faithfulness luôn = 1.0 | Prompt tách claim ra claim quá to, không nguyên tử | Ép "mỗi claim một sự kiện", cho ví dụ few-shot |
| Điểm nhảy loạn giữa 2 lần chạy | Judge dùng `temperature` > 0 | Đặt `temperature=0.0`, và đọc variance ở practice 07 |
| NDCG > 1.0 | Chia cho IDCG tính sai khi số relevant < k | IDCG phải dùng `min(len(relevant), k)` |
| MLflow không thấy run | Chưa `set_tracking_uri` hoặc sai đường dẫn sqlite | `mlflow.set_tracking_uri("sqlite:///mlflow.db")` — đường dẫn tương đối theo CWD |
| Eval chạy 40 phút | Gọi judge tuần tự, không cache | Bật `disk_cache`, và dùng `ThreadPoolExecutor(max_workers=4)` |
| Rate limit 429 từ OpenRouter | Gọi song song quá nhiều | Giảm `max_workers`, thêm backoff |

---

## Definition of Done

Đánh dấu khi thật sự làm được, không phải khi "đã đọc qua".

- [ ] Chạy hết 7 file practice, hiểu vì sao mỗi demo in ra như vậy.
- [ ] `eval_data/golden_set.json` tồn tại, **≥ 50 câu** đã qua validator.
- [ ] Báo cáo validator cho biết bao nhiêu câu bị loại và vì sao.
- [ ] `src/eval/` có đủ 5 module, import được từ ngoài.
- [ ] Chạy `run_eval` với 2 cấu hình khác nhau (vd `top_k=3` và `top_k=5`) trong
      **cùng một process** → ra hai kết quả **khác nhau**. (Nếu giống hệt: singleton.)
- [ ] MLflow UI hiện ≥ 2 run, so sánh được, có artifact `per_query.json`.
- [ ] Biết được σ của judge và ngưỡng khác biệt tối thiểu có ý nghĩa.
- [ ] Điền được dòng Baseline vào bảng kết quả ở `README.md`.

**Câu cho CV** (điền số thật của bạn):

> Xây evaluation harness cho hệ thống RAG: golden dataset 50+ câu tự sinh và
> kiểm định tự động (loại 4 lớp lỗi), 5 retrieval metric và 2 generation metric
> tự cài (LLM-as-judge có JSON schema, retry, cache), toàn bộ log vào MLflow.
> Đo variance của judge để xác định ngưỡng khác biệt có ý nghĩa (σ = 0.__).

---

## Câu hỏi phỏng vấn

**1. Bạn đánh giá RAG thế nào?**
Tách hai tầng. Retrieval đo Hit Rate/MRR/NDCG@k — xác định, không cần LLM, chạy được
mỗi lần commit. Generation đo Faithfulness và Answer Relevancy bằng LLM-as-judge.
Tách ra để khi điểm tụt biết tụt ở tầng nào; một điểm tổng thì không debug được.

**2. Faithfulness tính ra sao?**
Hai bước. Tách câu trả lời thành claim nguyên tử bằng LLM, rồi với từng claim hỏi
context có chống lưng không. Faithfulness = tỉ lệ claim được chống lưng. Chi phí là
1 + N lần gọi LLM mỗi câu, nên phải cache.

**3. Answer Relevancy không có ground truth thì đo kiểu gì?**
Sinh ngược: cho LLM đọc câu trả lời rồi sinh N câu hỏi mà nó trả lời được, sau đó đo
cosine giữa câu hỏi gốc và N câu sinh ra. Lạc đề thì câu sinh ngược khác xa câu gốc.
Ưu điểm là chạy được trên production traffic — nơi không bao giờ có ground truth.

**4. Sao không dùng RAGAS?**
Production thì dùng được. Ở đây tự viết để kiểm soát schema, cache và chi phí, và để
hiểu cơ chế. Thực tế RAGAS cũng làm đúng hai bước như trên.

**5. Làm sao biết judge của bạn đúng?**
Không có "đúng" tuyệt đối, chỉ có tin cậy được đến đâu. Tôi chấm cùng một câu 8 lần
để đo σ, thử độ nhạy khi đổi rubric, và dùng judge model khác nhà cung cấp với
generator để tránh self-preference bias. Từ σ tôi suy ra khác biệt tối thiểu có
ý nghĩa — dưới ngưỡng đó tôi không kết luận A tốt hơn B.

**6. Golden set sinh bằng LLM thì có tự lừa mình không?**
Có, nếu không lọc. Tôi loại 4 lớp: câu tự tham chiếu ("theo đoạn trên"), câu leak từ
vựng (overlap > 75% với chunk), câu trùng nhau (cosine > 0.93), và câu mà nhãn không
nằm trong top-5 khi truy hồi thử. Thường 100 câu sinh ra còn khoảng 55 câu dùng được.

**7. 50 câu có đủ không?**
Đủ để thấy chênh lệch lớn (> 7%), không đủ cho chênh lệch nhỏ. Nên ở Phase 3 tôi
dùng bootstrap CI: nếu khoảng tin cậy của hiệu số chứa 0 thì tôi báo "không kết luận
được", chứ không báo cấu hình thắng.

---

← [Phase 0 — Setup](00_setup.md) · [Phase 2 — Prompt Optimization](02_prompt_optimization.md) →
