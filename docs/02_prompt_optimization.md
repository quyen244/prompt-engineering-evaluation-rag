# Phase 2 — Prompt Optimization

> Phase này chỉ có nghĩa nếu Phase 1 đã xong. Không có thước đo thì "tối ưu
> prompt" chỉ là đổi chữ rồi tin là tốt hơn.

---

## Mục tiêu

1. Prompt là **artifact có version** trong MLflow Prompt Registry, load theo alias.
2. Bảng so sánh **≥ 5 biến thể** với số liệu nhiều chiều.
3. **Few-shot động** — chọn ví dụ theo câu hỏi, không viết cứng.
4. **Trích dẫn nguồn** ép được và **đo được độ chính xác của trích dẫn**.
5. **Vòng lặp tự tối ưu** evaluate → mutate → select, chạy ≥ 3 vòng.
6. **Train/test split** — chứng minh cải thiện không phải overfit.

---

## Nền tảng lý thuyết

### 1. Prompt trong code là nợ kỹ thuật

Ba vấn đề, cả ba chỉ lộ ra khi đã muộn:

- **Sửa prompt = sửa code = deploy.** Người viết prompt giỏi nhất trong team
  thường không có quyền merge. Quy trình sai từ gốc.
- **Không biết câu trả lời cũ sinh từ prompt nào.** User báo "hôm qua khác hôm
  nay", bạn `git log` và đoán.
- **Không so sánh được.** "Prompt mới tốt hơn" — hơn cái nào, đo trên gì?

MLflow Prompt Registry: prompt có tên, số version, alias, commit message, và
**gắn được vào run eval**.

```python
mlflow.genai.register_prompt(name="rag-answer", template=..., commit_message=...)
mlflow.genai.set_prompt_alias("rag-answer", alias="production", version=2)
prompt = mlflow.genai.load_prompt("prompts:/rag-answer@production")
prompt.format(context=..., question=...)
```

Cú pháp biến là `{{question}}` — **hai** ngoặc nhọn, để phân biệt với f-string.
`prompt.variables` là hợp đồng: quên truyền `context` thì lỗi ngay, thay vì gửi
lên LLM một prompt còn nguyên chữ `{{context}}` và nhận về câu trả lời vô nghĩa
một cách im lặng.

Tư duy giống hệt `@champion` của model ở Phase 4: **tách "cái gì đang chạy" khỏi
"code chạy nó"**.

### 2. Chấm nhiều chiều, không một điểm

| Chỉ số | Đo gì | Hỏng kiểu gì |
|---|---|---|
| `faithfulness` | Có bám ngữ cảnh không | Bịa thông tin |
| `refusal_ok` | Trả lời được thì trả lời, không thì từ chối | Bịa khi không biết / từ chối oan |
| `citation` | Trích đúng đoạn nguồn | Trích dẫn sai → niềm tin giả |
| `length` | Độ dài | Dài lê thê = tốn tiền |

Một prompt có thể **tăng faithfulness nhưng làm hỏng khả năng từ chối**. Bảng
phải cho thấy đánh đổi đó.

Trọng số của điểm tổng **không trung lập**: bài này dùng
`0.45×faithful + 0.35×refusal + 0.20×citation` vì bịa đặt là lỗi tệ nhất của RAG.
Bạn phải giải thích được lựa chọn đó khi ai hỏi.

### 3. Câu KHÔNG trả lời được là phần quan trọng nhất

Bộ dữ liệu Phase 2 có **3/10 câu** mà ngữ cảnh không chứa câu trả lời. Prompt kém
sẽ bịa — và **câu bịa trông tự tin, trôi chảy y hệt câu đúng**. Không cách nào
phân biệt bằng mắt ở quy mô lớn.

Nếu golden set của bạn chỉ có câu trả lời được, bạn đang không đo thứ nguy hiểm nhất.

### 4. Hai kết quả phản trực giác

**Chain-of-thought không phải thuốc tiên.** CoT giúp nhiều với bài toán suy luận
nhiều bước. RAG phần lớn là **trích xuất**: tìm đoạn đúng rồi diễn đạt lại. Suy
luận từng bước ở đây thường chỉ tốn token và latency.

**Nhiều quy tắc hơn ≠ tốt hơn.** Prompt 15 quy tắc thường thua prompt 4 quy tắc:

- Quy tắc mâu thuẫn ngầm ("ngắn gọn" vs "đầy đủ" vs "nêu mâu thuẫn").
- Chỉ dẫn ở giữa prompt dài bị chú ý ít hơn (*lost in the middle*).
- Model dành "ngân sách chú ý" cho việc tuân thủ format thay vì cho nội dung.

Bài học: thêm quy tắc phải **đo**. Trực giác "nói rõ hơn thì tốt hơn" sai thường
xuyên hơn bạn nghĩ.

### 5. Few-shot động > few-shot cố định

Few-shot cố định: ví dụ không liên quan làm nhiễu, và tốn token cho **mọi** request.

Few-shot động: ngân hàng ví dụ + chọn k cái gần nhất theo embedding. Dùng lại
chính embedding model của RAG, không cần model mới.

Ba chi tiết:

- **Nhúng theo câu HỎI, không theo câu trả lời.** Lúc chọn ví dụ bạn chưa có câu
  trả lời. Nhầm là bug im lặng.
- **Ngân hàng PHẢI có ví dụ TỪ CHỐI.** Nếu mọi ví dụ đều trả lời được, model học
  rằng "luôn phải trả lời" → tỉ lệ bịa **tăng**. Đây là tác hại ngược của few-shot
  mà rất ít người để ý.
- **MMR chống trùng lặp**: `λ·sim(ví dụ, câu hỏi) − (1−λ)·max sim(ví dụ, đã chọn)`.
  Ba ví dụ gần như giống nhau tốn token mà không dạy thêm gì.

### 6. Trích dẫn: ép thì dễ, đúng thì khó

Model **rất giỏi tạo trích dẫn trông như thật** — nó viết `[2]` vì prompt bảo thế,
không phải vì câu đó đến từ đoạn 2.

Bốn chỉ số hỏng theo bốn kiểu:

| Chỉ số | Nghĩa |
|---|---|
| `citation_rate` | Bao nhiêu % câu trả lời **có** trích dẫn (dễ nhất) |
| `citation_valid` | `[n]` có trỏ tới đoạn **tồn tại** không (model bịa `[7]` khi chỉ có `[1][2][3]`) |
| `precision` | Trong các đoạn đã trích, bao nhiêu **đúng** |
| `recall` | Trong các đoạn đúng, bao nhiêu **được trích** |

**Trích dẫn sai TỆ HƠN không trích dẫn.** `rate = 100%` với `precision = 40%` nguy
hiểm hơn hệ thống không trích dẫn gì — vì user nhìn thấy `[2]` thì tin và không
bấm vào kiểm tra. Trích dẫn tạo ra thẩm quyền; nếu thẩm quyền đó không có thật,
bạn đang làm hại người dùng.

Một lỗi kinh điển: *"Tài liệu không chứa thông tin này [1][2]"* — sinh ra khi
prompt vừa nói "LUÔN trích dẫn" vừa nói "không biết thì từ chối". Cách sửa: nêu
ngoại lệ **ngay trong** quy tắc trích dẫn, đừng để nó ở một quy tắc riêng phía dưới.

### 7. Vòng lặp tự tối ưu — và giới hạn của nó

```
   ┌─────────────────────────────────────────┐
   │ 1. CHẤM prompt hiện tại trên TRAIN      │
   │ 2. THU THẬP câu làm sai + LÝ DO         │
   │ 3. LLM đọc lỗi → SINH prompt mới        │
   │ 4. CHẤM prompt mới                      │
   │ 5. GIỮ cái TỐT NHẤT, lặp lại            │
   └─────────────────────────────────────────┘
                     ↓
        BÁO CÁO trên TEST (chưa từng nhìn)
```

Bốn nguyên tắc:

1. **Feedback cụ thể, không phải điểm số.** "Điểm 0.62" thì LLM không sửa được gì.
   Đưa câu hỏi cụ thể đã sai, câu trả lời sai, và lý do sai.
2. **Overfit là chắc chắn, không phải rủi ro.** Thử 20 biến thể rồi chọn cái tốt
   nhất trên 6 câu train — đó là *định nghĩa* của overfit.
3. **Giữ best-so-far, không phải cái mới nhất.** Vòng lặp đi xuống được.
4. **Biết dừng.** Cải thiện dưới ngưỡng nhiễu của judge (Phase 1 bài 7) thì dừng.

**Giới hạn quan trọng nhất — reward hacking:** vòng lặp tối ưu **chính xác** thứ
bạn đo, không hơn. Không đo độ dài → prompt sinh câu trả lời lê thê. Không đo
latency → nó thêm chain-of-thought. Không đo tỉ lệ từ chối → nó dạy model luôn
trả lời.

**Chất lượng vòng lặp bị chặn trên bởi chất lượng eval harness.** Đó là lý do
Phase 1 phải đứng trước Phase 2.

Mẹo thực dụng: dùng model **mạnh** (`JUDGE_MODEL`) để **sinh** prompt, model **rẻ**
(`GEN_MODEL`) để **chạy** prompt. Sinh xảy ra vài chục lần, chạy xảy ra hàng triệu lần.

---

## Thứ tự chạy practice

| # | File | Học gì | API |
|---|---|---|---|
| — | `prompt_common.py` | Bộ dữ liệu 10 câu (3 câu không trả lời được) + hàm chấm | — |
| 01 | `01_prompt_registry.py` | Đăng ký, alias, template variable, gắn vào run | **0** |
| 02 | `02_prompt_variants.py` | 5 biến thể, CoT có giúp không, nhiều quy tắc có tốt hơn không | ~100 |
| 03 | `03_few_shot_dynamic.py` | Top-k vs MMR, vì sao cần ví dụ TỪ CHỐI | ~60 |
| 04 | `04_citation_prompt.py` | 4 mức ép trích dẫn, 4 chỉ số chất lượng trích dẫn | ~80 |
| 05 | `05_auto_optimize.py` | Vòng lặp mutate-evaluate-select, train/test, reward hacking | ~250 |

```powershell
python practice/02_prompt_opt/01_prompt_registry.py
python practice/02_prompt_opt/02_prompt_variants.py
python practice/02_prompt_opt/03_few_shot_dynamic.py
python practice/02_prompt_opt/04_citation_prompt.py
python practice/02_prompt_opt/05_auto_optimize.py --rounds 3 --variants 2
```

Toàn bộ generation và judge đều cache trên đĩa (`.cache/`). Chạy lần hai gần như
miễn phí. Bỏ cache: `$env:NO_CACHE=1`.

---

## Ghép vào hệ thống

### Cấu trúc cần tạo

```
src/prompts/
  __init__.py
  registry.py          ← load theo alias, cache, fallback về file local
  templates/           ← bản .txt để đọc/review trong PR
    rag_answer_v3.txt
    judge_faithfulness.txt
  few_shot.py          ← ExampleBank, select_topk, select_mmr
  optimize.py          ← vòng lặp mutate-evaluate-select
```

### Signature nên có

```python
# src/prompts/registry.py
@dataclass(frozen=True)
class PromptRef:
    name: str
    version: int
    template: str
    variables: set[str]

def load(name: str, alias: str = "production") -> PromptRef: ...
def register(name: str, template: str, commit_message: str,
             tags: dict | None = None) -> int: ...
def promote(name: str, version: int, alias: str = "production") -> None: ...

# src/prompts/few_shot.py
class ExampleBank:
    def __init__(self, examples: list[Example], embedding: str | None = None): ...
    def build(self) -> None: ...                    # nhúng sẵn lúc startup
    def select(self, question: str, k: int = 3,
               strategy: str = "mmr") -> list[Example]: ...

# src/prompts/optimize.py
def optimize(seed: str, train: list[GoldenCase], test: list[GoldenCase],
             rounds: int = 3, variants: int = 2) -> OptimizeReport: ...
```

### Nguyên tắc thiết kế phải giữ

1. **Không hardcode prompt ở bất kỳ đâu trong `src/`.** Kể cả prompt của judge,
   của query rewriting, của summarization — thường nhiều hơn bạn nghĩ.
2. **Registry chết không được làm chết API.** Fallback về `templates/*.txt`, log
   cảnh báo. Cùng nguyên tắc với model registry ở Phase 4.
3. **Cache prompt đã load.** Đừng gọi registry mỗi request.
4. **Ngân hàng ví dụ nhúng sẵn lúc startup**, không nhúng mỗi request. Ngân hàng
   đổi hiếm; câu hỏi đến liên tục.
5. **`prompt_version` phải nằm trong response và trong log.** Cùng lý do với
   `model_version` ở Phase 4 — không có nó thì không điều tra được.
6. **Số đi vào CV phải là số trên test set.**

---

## Bẫy thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Prompt gửi đi còn nguyên `{{context}}` | Dùng `.format()` của Python trên template MLflow | Dùng `prompt.format()` của `PromptRef` |
| Điểm nhảy giữa hai lần chạy | `temperature > 0` khi chấm | `temperature=0.0` ở mọi chỗ trừ bước mutate |
| Prompt 15 quy tắc tệ hơn prompt 4 quy tắc | Quy tắc mâu thuẫn / lost in the middle | Cắt bớt, đo lại |
| *"Tài liệu không chứa thông tin này [1][2]"* | Quy tắc trích dẫn mâu thuẫn quy tắc từ chối | Nêu ngoại lệ ngay trong quy tắc trích dẫn |
| Few-shot làm tăng tỉ lệ bịa | Ngân hàng không có ví dụ TỪ CHỐI | Thêm ví dụ từ chối theo đúng tỉ lệ thật |
| Vòng auto-optimize sinh prompt thiếu placeholder | Không validate output | `extract_prompt()` kiểm tra `{context}`/`{question}` |
| Cải thiện train lớn, test = 0 | Overfit | Bình thường với tập nhỏ. **Báo cáo trung thực**, tăng cỡ mẫu |
| Câu trả lời dài dần qua mỗi vòng | Reward hacking — không đo độ dài | Thêm `length` vào hàm mục tiêu với trọng số âm |
| Prompt tốt trên model A, tệ trên model B | Prompt bám đặc tính model | Tối ưu lại khi đổi model; ghi `model_family` vào tag |

---

## Definition of Done

- [ ] Prompt lưu trong MLflow Prompt Registry, code load theo `@production`.
- [ ] Bảng so sánh **≥ 5 biến thể**, nhiều chiều, không phải một điểm.
- [ ] Chỉ ra được **ít nhất một** trực giác của bạn bị số liệu bác bỏ.
- [ ] Few-shot động chạy được, và **chứng minh được** ví dụ TỪ CHỐI có tác dụng
      (chạy có và không có nó, so cột `refusal_ok`).
- [ ] Đo được `citation_precision`, không chỉ `citation_rate`.
- [ ] Vòng auto-optimize chạy ≥ 3 vòng, cải thiện được metric trên train.
- [ ] **Báo cáo số trên TEST set**, và nói rõ khoảng cách train − test.
- [ ] Prompt thắng cuộc đã gắn alias `@production`.

**Câu cho CV** (điền số thật):

> Xây vòng lặp tự động tối ưu prompt (evaluate → mutate → select) dùng LLM đọc
> lỗi cụ thể để sinh biến thể mới; faithfulness tăng 0.__ → 0.__ trên **held-out
> test set** (chênh train−test 0.__). Prompt quản lý bằng MLflow Prompt Registry
> với alias `@production` — đổi prompt không cần deploy. Few-shot chọn động theo
> MMR, có ví dụ từ chối để giảm tỉ lệ bịa.

---

## Câu hỏi phỏng vấn

**1. Bạn quản lý prompt thế nào?**
Prompt là artifact có version trong MLflow Prompt Registry, không phải string
trong code. Code load theo alias `@production`. Đổi prompt là đổi alias, không
deploy. Mỗi run eval log `prompt_version`, nên mọi con số đều truy ngược được về
đúng prompt đã sinh ra nó.

**2. Làm sao biết prompt mới tốt hơn?**
Chấm bằng harness của Phase 1 trên nhiều chiều — faithfulness, tỉ lệ từ chối
đúng, độ chính xác trích dẫn, độ dài — chứ không một điểm tổng. Và số báo cáo
phải trên test set tách riêng khỏi tập dùng để chọn prompt.

**3. Chain-of-thought có giúp RAG không?**
Trong đo đạc của tôi thì không đáng kể. CoT mạnh với bài toán suy luận nhiều
bước; RAG phần lớn là trích xuất — tìm đoạn đúng rồi diễn đạt lại. CoT ở đây tăng
token và latency mà cải thiện nằm trong nhiễu.

**4. Vòng auto-optimize của bạn hoạt động ra sao?**
Chấm prompt trên train, thu thập các câu làm sai kèm lý do cụ thể, đưa cho một
model mạnh hơn đọc rồi sinh prompt mới, chấm lại, giữ cái tốt nhất. Dừng khi cải
thiện nhỏ hơn ngưỡng nhiễu của judge. Điểm mấu chốt là feedback phải là **lỗi cụ
thể**, không phải điểm số — điểm số thì LLM không biết sửa gì.

**5. Vòng lặp đó có overfit không?**
Chắc chắn có — thử hàng chục biến thể rồi chọn cái tốt nhất trên tập train thì
theo định nghĩa là overfit. Nên tôi tách test set và báo cáo cả khoảng cách
train−test. Nếu tỉ lệ chuyển giao gần 0 thì tôi nói thẳng là vòng lặp chỉ đang
ghi nhớ đáp án.

**6. Rủi ro lớn nhất của tự động tối ưu prompt là gì?**
Reward hacking. Vòng lặp tối ưu chính xác thứ tôi đo. Không đo độ dài thì nó sinh
câu trả lời lê thê; không đo tỉ lệ từ chối thì nó dạy model luôn trả lời. Chất
lượng vòng lặp bị chặn trên bởi chất lượng eval harness — đó là lý do tôi làm
Phase 1 trước.

**7. Vì sao few-shot cần ví dụ "từ chối"?**
Nếu mọi ví dụ đều là câu trả lời được, model học được mẫu "luôn phải có câu trả
lời" và tỉ lệ bịa tăng lên. Tôi giữ tỉ lệ ví dụ từ chối trong ngân hàng gần với
tỉ lệ thật trong dữ liệu.

**8. Trích dẫn nguồn — bạn đo thế nào?**
Bốn chỉ số tách riêng: tỉ lệ có trích dẫn, tỉ lệ trích dẫn hợp lệ (trỏ tới đoạn
tồn tại), precision và recall so với đoạn đúng. Chỉ đo tỉ lệ có trích dẫn là nguy
hiểm: một hệ thống trích dẫn 100% với precision 40% còn tệ hơn không trích dẫn,
vì user tin vào nó mà không kiểm tra.

---

← [Phase 1 — Evaluation](01_evaluation.md) · [Phase 3 — A/B Testing](03_ab_testing.md) →
