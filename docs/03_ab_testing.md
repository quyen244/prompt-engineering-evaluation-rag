# Phase 3 — A/B Testing

> Phase 1 xây cái cân. Phase này dùng nó để cân, và quan trọng hơn — để biết
> khi nào KHÔNG được kết luận.

---

## Mục tiêu

Kết thúc phase này bạn có:

1. Một **experiment matrix** thay đổi được chunker, embedding, retriever, top_k.
2. Kết quả **sweep có kiểm soát**, mỗi run log đầy đủ param + metric + per-query artifact.
3. **Bootstrap confidence interval** cho hiệu số giữa hai cấu hình.
4. Một **Pareto front** trên 3 trục: chất lượng × độ trễ × chi phí.
5. Một **cấu hình champion** chọn có lý do, viết ra được thành đoạn văn.

Đây là phase tạo ra bảng số cho CV. Nhưng thứ gây ấn tượng không phải bảng —
mà là việc bạn biết dòng nào trong bảng là thật và dòng nào là nhiễu.

---

## Nền tảng lý thuyết

### 1. OFAT trước, grid sau

**OFAT** (One Factor At A Time) = đổi đúng một biến, giữ nguyên phần còn lại.

```
baseline:  sentence / medium / bge-small / dense / k=5
   ├─ đổi chunker:    token, semantic, sentence-window
   ├─ đổi embedding:  bge-base, minilm, e5-multilingual
   ├─ đổi retriever:  bm25, hybrid, rerank
   └─ đổi top_k:      3, 10
```

12 run thay vì 4×4×4×3 = 192 run của grid đầy đủ.

| | OFAT | Grid search |
|---|---|---|
| Số run | tuyến tính | nhân |
| Cho biết | ảnh hưởng của từng biến riêng | cả tương tác giữa các biến |
| Điểm yếu | **bỏ sót tương tác** | tốn thời gian theo cấp số nhân |

Ví dụ tương tác thật: `semantic chunker` một mình chỉ hơn 1%, `bge-base` một mình
hơn 2%, nhưng ghép hai cái lại hơn 6% — vì chunk ngữ nghĩa dài hơn và chỉ model
mạnh hơn mới tận dụng được. OFAT không bao giờ thấy điều này.

**Quy trình đúng:** OFAT trước để tìm 2-3 biến có ảnh hưởng lớn nhất, rồi grid
search chỉ trên những biến đó. `04_grid_search.py` hỗ trợ cả `--strategy ofat`
và `--strategy grid`.

### 2. Điều gì phải giữ cố định

Một A/B test sai khi có biến trôi mà bạn không biết. Danh sách phải khoá:

- **Golden set** — cùng file, cùng số câu. Sinh lại golden set giữa chừng là hỏng.
- **Judge model + judge prompt** — đổi giám khảo thì mọi điểm cũ vô giá trị.
- **Seed** — cho mọi bước có ngẫu nhiên (sampling, shuffle).
- **Corpus** — thêm tài liệu vào `data/` giữa sweep là tự phá thí nghiệm.
- **Git commit** — `ab_common.git_commit()` log nó vào mọi run. Nhìn lại 2 tuần sau
  bạn sẽ cảm ơn chính mình.

### 3. Đo latency cho đúng

Ba lỗi kinh điển:

| Lỗi | Hậu quả | Cách sửa |
|---|---|---|
| Báo cáo trung bình | Trung bình che mất đuôi. User cảm nhận đuôi. | Báo **p50 và p95** |
| Không warm-up | Lần gọi đầu tính cả thời gian load model | Chạy 1 query bỏ đi trước khi đo |
| Trộn build-time vào query-time | Cấu hình index chậm trông như query chậm | Đo `build_seconds` riêng |

`ab_common.percentiles()` làm phần p50/p95. `04_grid_search.py` tách `build_s`.

### 4. Bootstrap CI — công cụ chống tự lừa mình

Bạn đo cấu hình A: NDCG 0.784. Cấu hình B: NDCG 0.802. B thắng?

**Chưa biết.** 50 câu hỏi là một mẫu. Đổi 50 câu khác thì thứ tự có thể lật.

Bootstrap trả lời chính xác câu đó:

```
lặp 5000 lần:
    lấy mẫu lại 50 câu CÓ HOÀN LẠI từ golden set
    tính NDCG_B − NDCG_A trên mẫu đó
→ có 5000 giá trị hiệu số
→ lấy percentile 2.5% và 97.5% → khoảng tin cậy 95%
```

Đọc kết quả:

| CI của (B − A) | Kết luận |
|---|---|
| `[+0.012, +0.061]` | B thắng thật. Toàn bộ khoảng > 0. |
| `[−0.019, +0.055]` | **Không kết luận được.** Khoảng chứa 0. |
| `[−0.048, −0.005]` | A thắng thật. |

Điểm mấu chốt: phải **lấy mẫu lại theo câu hỏi**, dùng **cùng bộ câu hỏi** cho cả
A và B trong mỗi vòng lặp (paired bootstrap). Nếu lấy mẫu độc lập, khoảng tin cậy
sẽ rộng ra vô lý và bạn sẽ không bao giờ phát hiện được cải thiện thật.

`05_significance.py` cài đủ, thêm hàm `required_n()` cho biết cần bao nhiêu câu
để thu khoảng tin cậy xuống mức mong muốn.

### 5. Pareto front — chọn champion trên nhiều trục

Sắp theo NDCG rồi lấy dòng đầu = deploy hệ thống chậm gấp 3 để đổi lấy nhiễu.

- **Dominance**: A trội B khi A không tệ hơn ở mọi trục và tốt hơn ở ít nhất một.
- **Pareto front**: tập không bị ai trội. 25 cấu hình thường còn 4-6.
- **SLO là bộ lọc cứng, áp TRƯỚC** khi tính front — không phải một trục để mặc cả.
- **Knee point**: chỗ đường cong bẻ gãy, "trả nhiều được ít" bắt đầu từ đó.

Sức mạnh của Pareto: **không cần gán trọng số**. Bạn không bao giờ phải trả lời
câu hỏi bịa "1 điểm NDCG đáng bao nhiêu ms?".

`06_pareto_plot.py` in ASCII plot và ghi `artifacts/pareto_front.png`.

### 6. Hybrid retrieval và RRF

Dense embedding giỏi ngữ nghĩa, dốt từ khoá hiếm (mã lỗi, tên riêng, số hiệu).
BM25 thì ngược lại. Ghép bằng **Reciprocal Rank Fusion**:

```
RRF_score(d) = Σ  1 / (k + rank_i(d))        với k = 60 theo mặc định
             i∈{dense, bm25}
```

Vì sao dùng rank chứ không dùng score? Vì cosine của dense và điểm BM25 **không
cùng thang đo** và không chuẩn hoá được một cách có nguyên tắc. Rank thì luôn so
được. Đây là câu hỏi phỏng vấn hay gặp.

Tầng cuối là **cross-encoder rerank**: lấy top-50 từ hybrid, cho cross-encoder đọc
từng cặp (query, chunk) rồi xếp lại, giữ top-5. Chính xác nhất, chậm nhất — và
`06_pareto_plot.py` sẽ cho bạn thấy nó thường nằm ngoài SLO.

---

## Thứ tự chạy practice

| # | File | Học gì | Tốn API? |
|---|---|---|---|
| — | `ab_common.py` | Hạ tầng chung: corpus, chunker, DenseIndex, BM25Index, RRF, reranker, đo lường | — |
| 01 | `01_chunker_ab.py` | 4 chunker × 3 mức kích thước, ảnh hưởng lên retrieval | **0** |
| 02 | `02_embedding_ab.py` | 4 embedding model, chất lượng vs tốc độ vs RAM | **0** |
| 03 | `03_retriever_ab.py` | dense vs bm25 vs hybrid(RRF) vs rerank | **0** |
| 04 | `04_grid_search.py` | Sweep có kiểm soát, log MLflow nested run | **0** |
| 05 | `05_significance.py` | Paired bootstrap CI, cỡ mẫu cần thiết | **0** |
| 06 | `06_pareto_plot.py` | Dominance, front, SLO, knee point, biểu đồ | **0** |

Cả phase **không tốn một đồng API nào** — vì embedding chạy local và metric
retrieval không cần LLM. Đây chính là lý do Phase 0 chọn embedding local.

```powershell
python practice/03_ab_testing/01_chunker_ab.py
python practice/03_ab_testing/02_embedding_ab.py
python practice/03_ab_testing/03_retriever_ab.py
python practice/03_ab_testing/04_grid_search.py --strategy ofat
python practice/03_ab_testing/05_significance.py
python practice/03_ab_testing/06_pareto_plot.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`06` chạy được ngay với `--demo` nếu bạn chưa chạy sweep.

**Lưu ý về corpus.** `ab_common.load_corpus()` yêu cầu tối thiểu 20.000 ký tự.
Với corpus nhỏ hơn, khác biệt giữa các cấu hình sẽ chìm trong sai số lấy mẫu và
mọi kết luận đều vô nghĩa.

---

## Ghép vào hệ thống

### Cấu trúc cần tạo

```
src/experiments/
  __init__.py
  space.py         ← định nghĩa không gian cấu hình + sinh danh sách run
  runner.py        ← chạy một cấu hình, trả EvalReport (gọi src/eval/runner.py)
  analysis.py      ← bootstrap CI, dominance, pareto_front, knee_point
  registry.py      ← ghi champion vào MLflow, gắn alias
```

### Signature nên có

```python
# src/experiments/space.py
@dataclass(frozen=True)
class Axis:
    name: str
    values: list[Any]

class ExperimentSpace:
    def __init__(self, baseline: EvalConfig, axes: list[Axis]): ...
    def ofat(self) -> list[EvalConfig]: ...
    def grid(self, axes: list[str] | None = None) -> list[EvalConfig]: ...
    def random(self, n: int, seed: int) -> list[EvalConfig]: ...

# src/experiments/analysis.py
def paired_bootstrap_ci(a_scores: list[float], b_scores: list[float],
                        n_resamples: int = 5000, alpha: float = 0.05,
                        seed: int = 0) -> tuple[float, float]: ...
def dominates(a: dict, b: dict, objectives: dict[str, str]) -> bool: ...
def pareto_front(rows: list[dict], objectives: dict[str, str]) -> list[dict]: ...
def knee_point(front: list[dict]) -> dict | None: ...

# src/experiments/registry.py
def promote_champion(run_id: str, model_name: str) -> None: ...
```

### Nguyên tắc thiết kế phải giữ

1. **`ExperimentSpace` sinh `EvalConfig`, không sinh dict.** Kiểu dữ liệu chung với
   Phase 1 nghĩa là thêm một trục mới chỉ cần thêm một trường vào `EvalConfig`.
2. **Cache index theo `(chunker, level, embedding)`.** `top_k` và `retriever` đổi
   không cần build lại index. Grid 24 run có khi chỉ cần build 6 index — tiết kiệm
   80% thời gian. `04_grid_search.py` làm sẵn bằng `group_key()`.
3. **Luôn log per-query.** Không có nó thì không bootstrap được, và bạn sẽ phải
   chạy lại cả sweep.
4. **Nested run trong MLflow.** Một parent run cho cả sweep, mỗi cấu hình là child.
   Không có cấu trúc này thì 24 run phẳng trong UI sẽ không đọc nổi.
5. **Log param TRƯỚC khi chạy, metric SAU.** Sweep chết giữa chừng thì bạn vẫn
   biết run dở dang đó đang thử cấu hình gì.

### Sửa code cũ

| File | Việc cần làm |
|---|---|
| `src/rag/chunking.py` | Giữ nguyên `ChunkerFactory` — đúng hình dạng rồi. Bổ sung `SentenceWindowNodeParser` nếu chưa có. |
| `src/rag/rag_chatbot.py` | Retriever phải nhận được từ ngoài vào (dependency injection), không tự tạo bên trong. |
| `src/core/providers.py` | Cache embedding model theo tên — sweep 4 model không nên load lại model đã dùng. |

---

## Bẫy thường gặp

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Mọi cấu hình ra kết quả **giống hệt nhau** | Singleton — config thứ 2 trở đi bị bỏ qua | Bỏ `_chatbot_instance`, xem `docs/00_setup.md` |
| NDCG cao bất thường (> 0.95) ở mọi cấu hình | Golden set bị leak từ vựng | Chạy lại validator Phase 1 |
| Bootstrap CI rộng vô lý | Lấy mẫu độc lập thay vì **paired** | Cùng bộ câu hỏi cho A và B mỗi vòng lặp |
| p95 nhảy loạn giữa 2 lần chạy | Không warm-up, hoặc máy đang chạy việc khác | Warm-up 1 query, đóng ứng dụng nặng |
| Sweep chạy 3 tiếng | Build lại index cho mọi run | Nhóm theo `group_key()` |
| `semantic` chunker cực chậm | Nó gọi embedding cho từng câu để tìm điểm cắt | Bình thường. Ghi nhận `build_s` và để Pareto quyết. |
| Rerank tốt hơn hẳn nhưng p95 > 1s | Cross-encoder đọc từng cặp, không precompute được | Đúng như lý thuyết. Áp SLO rồi xem nó có sống sót không. |
| MLflow UI không nhóm nested run | Thiếu `nested=True` ở child run | `mlflow.start_run(nested=True)` |
| Kết luận đảo chiều khi chạy lại | Cỡ mẫu quá nhỏ | `required_n()` ở file 05 cho biết cần bao nhiêu câu |

---

## Definition of Done

- [ ] Chạy hết 6 file practice, đọc được mọi bảng chúng in ra.
- [ ] Chạy OFAT ≥ 12 cấu hình trên golden set thật của bạn.
- [ ] MLflow có 1 parent run + N child run, mỗi child có `per_question.json`.
- [ ] Với mỗi cấu hình so baseline: có khoảng tin cậy 95% của hiệu NDCG.
- [ ] Nói được **ít nhất một** cấu hình mà bạn *tưởng* thắng nhưng CI chứa 0.
- [ ] Có `artifacts/pareto_front.png` sinh từ số liệu thật.
- [ ] Chọn được champion và viết được 3-4 câu giải thích vì sao — kèm SLO và
      cấu hình bị loại.
- [ ] Điền cột Champion vào bảng kết quả ở `README.md`.

**Câu cho CV** (điền số thật của bạn):

> Chạy A/B test có kiểm soát trên __ cấu hình RAG (chunker × embedding × retriever
> × top-k) với MLflow tracking: NDCG@5 tăng từ 0.__ lên 0.__ (bootstrap 95% CI
> [+0.___, +0.___]), p95 latency __ms trong SLO 800ms. Chọn champion bằng Pareto
> front 3 trục thay vì tối ưu một chỉ số, loại __ cấu hình bị trội.

---

## Câu hỏi phỏng vấn

**1. Bạn A/B test RAG thế nào?**
OFAT trước để đo ảnh hưởng riêng của từng biến, rồi grid trên 2-3 biến mạnh nhất.
Mỗi run log param, metric và kết quả từng câu vào MLflow dưới dạng nested run.
Golden set, judge model và seed giữ cố định tuyệt đối trong cả sweep.

**2. Làm sao biết cải thiện là thật chứ không phải nhiễu?**
Paired bootstrap: lấy mẫu lại golden set có hoàn lại 5000 lần, mỗi lần tính hiệu
số giữa hai cấu hình trên **cùng** bộ câu hỏi, rồi lấy khoảng 2.5–97.5 percentile.
CI chứa 0 thì tôi báo "không kết luận được", không báo cấu hình thắng.

**3. Vì sao paired mà không phải hai mẫu độc lập?**
Vì cùng một câu hỏi khó thì cả A và B đều làm tệ — sai số đó tương quan. Paired
loại được phần biến thiên chung, cho khoảng tin cậy hẹp hơn nhiều với cùng số câu.

**4. Chọn champion thế nào khi cấu hình tốt nhất lại chậm nhất?**
Không tối ưu một chỉ số. Áp SLO làm bộ lọc cứng trước, rồi tính Pareto front trên
chất lượng × độ trễ × chi phí, rồi chọn knee point. Ưu điểm là không phải bịa
trọng số giữa NDCG và millisecond.

**5. Hybrid retrieval ghép điểm kiểu gì?**
Reciprocal Rank Fusion trên **thứ hạng**, không phải trên điểm số — vì cosine của
dense và điểm BM25 không cùng thang đo. RRF cộng 1/(60 + rank) từ mỗi retriever.

**6. Vì sao không dùng cross-encoder cho luôn?**
Cross-encoder đọc cặp (query, chunk) cùng lúc nên không precompute được — với 100k
chunk thì mỗi query phải chạy 100k lần forward. Nó chỉ dùng ở tầng rerank trên
top-50 đã lọc sẵn bằng bi-encoder.

**7. OFAT bỏ sót gì?**
Tương tác. Chunker ngữ nghĩa và embedding mạnh mỗi cái riêng lẻ chỉ hơn 1-2%, ghép
lại có thể hơn 6%. Nên OFAT chỉ để sàng lọc, kết luận cuối phải grid trên các biến
đã sàng ra.

---

← [Phase 2 — Prompt Optimization](02_prompt_optimization.md) · [Phase 4 — Serving](04_serving.md) →
