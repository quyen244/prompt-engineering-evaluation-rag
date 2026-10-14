# TechQA RRF tuning: 50 câu tune, 20 câu test chung

Thí nghiệm này so BM25, dense Qwen và hybrid Reciprocal Rank Fusion (RRF) trên cùng một corpus.
Chỉ 50 câu tune được dùng để chọn tham số RRF; 20 câu test được chấm sau khi tham số đã khóa.

## Dữ liệu và nguồn gốc

- Corpus gốc: `D:\Downloads\TechQA\TechQA\technote_corpus\technotes.id_title_text`.
- Câu hỏi train: 50 câu có đáp án, chọn bằng seed `20261014` từ các ID `TRAIN_Q...` trong [NVIDIA TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval), một bản đóng gói lại của [TechQA do IBM công bố](https://github.com/IBM/techqa).
- Câu hỏi test: 20 câu trong `eval_data/eval_questions.json` đã dùng ở thí nghiệm TechQA trước đó.
- Corpus chung: 150 tài liệu với 798 passage, gồm 100 document ID của bộ nhẹ hiện có và gold documents của 50 câu tune; text của cả 150 tài liệu được trích từ corpus gốc ở đường dẫn trên.
- Hai nhóm không trùng question ID hoặc gold document ID.

Đã kiểm tra cả 50 answer spans của nhóm tune xuất hiện nguyên văn trong text của gold document lấy từ corpus gốc.

Thư mục TechQA gốc trên máy hiện không có `training_Q_A.json` hoặc `dev_Q_A.json`; file `full_technote_collection.sections.json` cũng kết thúc giữa một JSON object.
Vì thế câu hỏi train lấy từ bản đóng gói NVIDIA, còn tài liệu dùng để lập index lấy từ file `technotes.id_title_text` của TechQA gốc.
File [manifest.json](../eval_data/rrf_study/manifest.json) lưu seed, ID và SHA-256 của các đầu vào và bộ dữ liệu đã tạo.

## Phương pháp

Tất cả phương pháp dùng passage 850 ký tự với overlap 150 và cùng 150 tài liệu.
Dense dùng Ollama `qwen3-embedding:0.6b`, digest `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`.
BM25 và dense xếp hạng toàn bộ passage một lần cho mỗi câu hỏi.
RRF cộng `1 / (rrf_k + rank)` của hai danh sách passage, sau đó mỗi tài liệu chỉ giữ passage có thứ hạng cao nhất.
Grid `rrf_k` là `1, 5, 10, 20, 40, 60, 100`.
Tiêu chí chọn trên 50 câu tune là nDCG@5, rồi Hit Rate@5, MRR@5 và giá trị `rrf_k` nhỏ hơn khi hòa.
Sau khi chọn `rrf_k`, cả BM25, dense và RRF được chấm trên cùng 20 câu test tại Top-K 1, 3, 5.
Các metric là retrieval theo gold document ID; chưa đo chất lượng câu trả lời hoặc faithfulness trong thí nghiệm này.

## Kết quả tune trên 50 câu

| RRF k | Hit@1 | Hit@3 | Hit@5 | MRR@5 | nDCG@5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9400 | 0.9800 | 0.9800 | 0.9600 | 0.9652 |
| 5 | 0.9400 | 0.9800 | 0.9800 | 0.9600 | 0.9652 |
| 10 | 0.9400 | 0.9800 | 0.9800 | 0.9600 | 0.9652 |
| 20 | 0.9200 | 0.9800 | 0.9800 | 0.9467 | 0.9552 |
| 40 | 0.9200 | 0.9600 | 0.9800 | 0.9450 | 0.9539 |
| 60 | 0.9200 | 0.9600 | 0.9800 | 0.9450 | 0.9539 |
| 100 | 0.9200 | 0.9600 | 0.9800 | 0.9450 | 0.9539 |

`rrf_k=1`, `5` và `10` hòa theo tiêu chí đã định; quy tắc phá hòa chọn `rrf_k=1`.

## Kết quả test chung trên 20 câu

| Phương pháp | Hit@1 | Hit@3 | Hit@5 | Precision@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.8000 | 0.8000 | 0.8500 | 0.1700 | 0.8125 | 0.8215 |
| Dense Qwen | **1.0000** | **1.0000** | **1.0000** | **0.2000** | **1.0000** | **1.0000** |
| RRF k=1 | 0.8000 | 1.0000 | 1.0000 | 0.2000 | 0.9000 | 0.9262 |

RRF đưa gold document xuống hạng 2 ở `TRAIN_Q306`, `TRAIN_Q121`, `DEV_Q132` và `TRAIN_Q272`; dense xếp cả bốn gold documents ở hạng 1.
RRF vẫn tìm đủ 20 gold documents trong Top-K 3, nhưng không cải thiện được dense ở Top-K 1 hoặc Top-K 5.
Vì thế phép thử này không ủng hộ việc đổi retrieval champion hiện tại sang RRF.

20 câu test này đã được dùng trong một thí nghiệm trước để chọn dense champion, nên không phải holdout hoàn toàn mới cho toàn bộ lịch sử project.
Chênh lệch trên 20 câu cũng chưa đủ để khẳng định kết quả sẽ giữ nguyên trên tập câu hỏi lớn hơn.

## Tái lập và kiểm tra MLflow

Chạy trong virtual environment Python 3.11 có `mlflow` và `httpx`; cần Ollama đang phục vụ `qwen3-embedding:0.6b`.
Nếu thiếu file câu hỏi train, tải lại bản đóng gói NVIDIA vào `data/techqa_rag_eval_train.json` trước khi chuẩn bị dữ liệu.

```powershell
Invoke-WebRequest -Uri 'https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval/resolve/main/train.json' `
  -OutFile 'data/techqa_rag_eval_train.json'
& .venv-loadtest/Scripts/python.exe -m scripts.prepare_techqa_rrf_study
& .venv-loadtest/Scripts/python.exe -m scripts.run_techqa_rrf_tuning
```

MLflow experiment là `techqa-rrf-tuning`; parent run `7fd833b2e13c42b095ed99e4f0d4da75` chứa provenance và bảng tổng hợp.
Mỗi ứng viên tune và mỗi phương pháp test có child run riêng, metric đầy đủ và artifact `evaluation/per_case.json` gồm ID, thứ hạng và điểm từng câu mà không lưu text câu hỏi hoặc câu trả lời.
Tóm tắt máy đọc được nằm ở `eval_data/results/techqa_rrf_tuning_summary.json`.
