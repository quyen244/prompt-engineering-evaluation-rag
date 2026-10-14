# TechQA vLLM inference benchmark

Ngày đo: 2026-10-15.
Toàn bộ số liệu trong tài liệu này đến từ endpoint vLLM đang chạy tại `http://127.0.0.1:8000` và các artifact MLflow của project.

## Dữ liệu, retrieval và server

- Bộ test gồm 20 câu TechQA đã dùng chung cho BM25, dense và RRF; 50 câu training riêng được dùng để tune RRF trước đó.
- Corpus nhẹ có 150 tài liệu và 798 passage, cắt theo 850 ký tự với overlap 150.
- Retrieval được chạy lại bằng dense `ollama:qwen3-embedding:0.6b`, TopK=1.
- Run MLflow `3e587bada3fe425d8c0462a454532f14` trong experiment `techqa-vllm-retrieval-verified` đạt Hit@1, MRR@1 và nDCG@1 đều bằng 1.000 trên 20 câu.
- Cả 20 document ID và chunk ID của lượt chạy lại trùng với artifact dense của lần so BM25/RRF trước đó.
- Generator là `Qwen/Qwen2.5-1.5B-Instruct` FP16 qua vLLM 0.31.0 trên WSL2 Ubuntu.
- GPU là NVIDIA GeForce GTX 1660 Ti 6144 MiB.
- Lệnh server đang chạy là `vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 2048 --gpu-memory-utilization 0.8 --host 127.0.0.1 --port 8000`.
- `/metrics` có `prefix_cache_queries_total` và `prefix_cache_hits_total` tăng theo request, xác nhận prefix cache có hoạt động trong cấu hình đang chạy.

## Cách đo

Script [vllm_techqa_benchmark.py](../scripts/vllm_techqa_benchmark.py) gọi OpenAI-compatible chat completions ở chế độ streaming.
TTFT được đo từ lúc gửi request đến chunk nội dung đầu tiên, còn latency được đo đến khi stream kết thúc.
TPOT được tính bằng thời gian từ token đầu đến lúc hoàn tất chia cho số output token còn lại, khi có ít nhất hai output token.
Request/s và output token/s dùng tổng số request hoàn tất và output token chia cho thời gian tường của batch.
Token count lấy từ `stream_options.include_usage` do vLLM trả về.
Queue time, prefill time, cache hit và KV cache usage lấy từ chênh lệch hoặc gauge của `/metrics`.
VRAM lấy từ `nvidia-smi`, là mức dùng của toàn GPU, bao gồm cả Ollama embedding nếu còn nạp trong bộ nhớ.
Mỗi scenario có một MLflow run riêng trong experiment `techqa-vllm-inference`, lưu model, version, cấu hình, SHA-256 dataset và ranking, metric cùng raw record không chứa question, context hoặc answer text.
Warmup không nằm trong batch đo; cold/warm của prefix được ghi thành hai run tách biệt.

## Baseline một request với cache được cách ly

Giả thuyết: input dài chủ yếu tăng TTFT qua prefill, còn output dài tăng thời gian hoàn thành qua decode.
Mỗi prompt có nonce riêng trong system message để tránh tái sử dụng prefix từ các scenario trước.
Mỗi dòng có 3 request đo, concurrency 1 và nhiệt độ model đã ổn định sau khi server hoạt động.

| Mục tiêu input | Input token TB | Output token TB | TTFT TB | TPOT TB | Latency TB | p95 latency | Output token/s | Cache hit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 ký tự | 255 | 32 | 1.44 s | 0.016 s | 1.95 s | 2.05 s | 16.44 | 14.6% |
| 1300 ký tự | 561 | 32 | 2.91 s | 0.017 s | 3.43 s | 3.64 s | 9.34 | 6.7% |
| 4500 ký tự | 1532 | 32 | 7.80 s | 0.018 s | 8.35 s | 9.35 s | 3.83 | 2.4% |
| 1300 ký tự | 564 | 122 | 2.93 s | 0.017 s | 5.00 s | 5.24 s | 24.49 | 6.6% |

Input dài làm TTFT tăng chủ yếu ở pha prefill, còn output dài làm latency hoàn tất tăng sau token đầu.
Các con số trên chỉ mô tả ba câu mẫu cho mỗi dòng, nên p95 ở đây chưa phải ước lượng tail latency ổn định.

## Prefix caching

Giả thuyết: đưa tài liệu chung lên trước câu hỏi giúp vLLM tái sử dụng prefix và giảm TTFT cho câu hỏi tiếp theo.
Mỗi cấu trúc có nonce mới để cold run không hưởng cache từ cấu trúc trước đó.
Mỗi cold run gồm 6 câu hỏi khác nhau, sau đó warm run lặp lại đúng 6 prompt này.
`doc_shared` đặt cùng tài liệu trước phần câu hỏi; `partial_shared` chia sẻ nửa đầu context; `reordered` thay đổi thứ tự các dòng tài liệu giữa các request.

| Cấu trúc | Cache hit cold | TTFT cold | Cache hit warm | TTFT warm |
|---|---:|---:|---:|---:|
| Unique | 4.1% | 5.46 s | 99.2% | 0.55 s |
| Chỉ chung system | 5.4% | 5.35 s | 99.4% | 0.50 s |
| Chung tài liệu | 75.9% | 2.17 s | 99.4% | 0.56 s |
| Chung một phần tài liệu | 51.8% | 3.26 s | 99.2% | 0.56 s |
| Đổi thứ tự tài liệu | 61.3% | 3.24 s | 99.4% | 0.55 s |

Trên cold run, đặt cùng tài liệu ở đầu prompt giảm TTFT trung bình khoảng 60.1% so với prompt unique có độ dài tương tự.
Đổi thứ tự làm mất một phần prefix trùng nhau và TTFT tăng trong mẫu này.
Warm run lặp lại nguyên prompt nên mức cache hit gần 99% ở mọi cấu trúc; kết quả warm không đại diện cho câu hỏi mới.
Server hiện không có API đổi prefix cache ở runtime, nên phép đo này không phải so sánh cache bật với cache tắt.
[Tài liệu vLLM về Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) mô tả cơ chế tái sử dụng KV khi token prefix trùng nhau.
Xem [biểu đồ TTFT cold và warm theo cấu trúc prefix](assets/vllm_prefix.svg).

## Context length và KV cache pressure

Giả thuyết: prompt dài hơn làm prefill và thời gian chờ tăng, đặc biệt khi có nhiều request đồng thời.
Nhóm này dùng nonce riêng cho từng request để cache hit thấp, với output tối đa 32 token.
Các mức input là mục tiêu ký tự, không phải số token cố định; số token và lỗi thực tế nằm trong raw record.

| Mục tiêu input | Concurrency | Thành công | TTFT TB | p95 latency | Request/s | Cache hit |
|---:|---:|---:|---:|---:|---:|---:|
| 700 ký tự | 1 | 2/2 | 2.24 s | 2.83 s | 0.363 | 8.3% |
| 700 ký tự | 4 | 8/8 | 5.82 s | 18.88 s | 0.224 | 11.4% |
| 1800 ký tự | 1 | 2/2 | 3.74 s | 4.28 s | 0.234 | 4.3% |
| 1800 ký tự | 4 | 8/8 | 10.29 s | 24.57 s | 0.173 | 6.3% |
| 3500 ký tự | 1 | 2/2 | 6.95 s | 7.78 s | 0.133 | 2.5% |
| 3500 ký tự | 4 | 8/8 | 16.56 s | 36.33 s | 0.119 | 3.7% |
| 6000 ký tự | 1 | 1/2 | 9.99 s | 10.56 s | 0.092 | 0.8% |
| 6000 ký tự | 4 | 5/8 | 19.56 s | 44.21 s | 0.096 | 2.5% |

Các request lỗi ở 6000 ký tự có prompt 2213, 2368 và 2408 token theo endpoint `/tokenize`, vượt `max_model_len=2048` trước khi sinh đáp án.
Giới hạn thực tế phải chừa chỗ cho chat template và output token; mức 4096 token không được phép đo trên server này.
GPU memory peak gần như đứng quanh 5152-5154 MiB trong các run này vì vLLM đã cấp phát KV cache từ trước, nên không thể suy lượng KV memory của từng context từ riêng `nvidia-smi`.
Không có đối chứng engine khác, vì vậy số liệu này không cô lập được lợi ích của PagedAttention.
Xem [biểu đồ TTFT theo độ dài context](assets/vllm_context.svg).

## Concurrency và continuous batching

Giả thuyết: batching tăng throughput khi có thêm request đồng thời nhưng cũng làm TTFT và tail latency tăng.
Mỗi request dùng tối đa 64 output token và prompt có nonce riêng để giới hạn tái sử dụng cache giữa các lượt.
Các mức 1, 2, 4 và 8 có 8 request mỗi mức; mức 16 có 16 request và mức 32 có 32 request.
Mức 2 được chạy lại độc lập và vẫn cho 0.083 request/s, nên đây là kết quả lặp lại của cấu hình hiện tại, dù cơ chế gây suy giảm chưa được xác định.

| Đồng thời | Thành công | Request/s | Output token/s | TTFT TB | Latency p50 | Latency p95 | Queue TB | VRAM peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 0.302 | 18.86 | 2.25 s | 3.55 s | 4.50 s | <0.01 s | 5110 MiB |
| 2 | 8/8 | 0.083 | 5.19 | 3.07 s | 24.39 s | 26.38 s | <0.01 s | 4981 MiB |
| 4 | 8/8 | 0.141 | 8.83 | 6.19 s | 28.14 s | 28.76 s | <0.01 s | 5103 MiB |
| 8 | 8/8 | 0.231 | 14.41 | 11.14 s | 34.69 s | 34.69 s | 0.24 s | 4981 MiB |
| 16 | 16/16 | 0.343 | 21.66 | 17.65 s | 46.37 s | 46.70 s | 1.30 s | 4982 MiB |
| 32 | 32/32 | 0.433 | 27.35 | 32.15 s | 73.15 s | 73.88 s | 11.24 s | 5074 MiB |

Throughput lớn nhất trong các mức đã thử là 0.433 request/s ở 32 request đồng thời, nhưng p95 latency gần 74 giây và queue trung bình hơn 11 giây.
Điểm này là giới hạn tải được đo, không phải throughput tối đa tuyệt đối của server vì throughput vẫn tăng đến mức cuối.
Mức 32 có p99 latency 73.89 giây trên đúng 32 mẫu, chưa đủ để suy ra tail latency dài hạn.
Nếu ưu tiên phản hồi tương tác, một request đồng thời có TTFT và latency thấp nhất trong phép đo.
Biểu đồ [throughput và latency](assets/vllm_concurrency.svg) cho thấy đánh đổi này.

## Workload composition và scheduling

Giả thuyết: input dài gây chậm prefill, còn output dài giữ tài nguyên decode lâu hơn và có thể làm chậm request ngắn trong batch hỗn hợp.
Bốn loại lần lượt là input khoảng 400 ký tự hoặc 4500 ký tự, với output tối đa 32 hoặc 128 token.
Mỗi loại có 8 request ở concurrency 4; lượt mixed có 20 request xen kẽ bốn loại và cũng ở concurrency 4.
Giá trị input token và output token trong bảng là trung bình thực tế của các request thành công.

| Loại | Input token TB | Output token TB | Thành công | TTFT TB | Latency TB | p95 latency | Request/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ngắn / ngắn | 307 | 32 | 8/8 | 4.21 s | 15.23 s | 16.18 s | 0.261 |
| Ngắn / dài | 307 | 113 | 8/8 | 3.90 s | 41.88 s | 49.93 s | 0.090 |
| Dài / ngắn | 1483 | 32 | 8/8 | 17.63 s | 35.06 s | 35.87 s | 0.114 |
| Dài / dài | 1424 | 116 | 7/8 | 13.63 s | 59.90 s | 70.33 s | 0.058 |
| Mixed | 800 | 68 | 19/20 | 6.38 s | 37.32 s | 76.13 s | 0.105 |

Trong mixed, nhóm ngắn / ngắn có latency trung bình 16.31 giây so với 15.23 giây khi chạy riêng, còn nhóm ngắn / dài có 65.65 giây so với 41.88 giây.
Đây là dấu hiệu request dài làm chậm request khác trên GPU này, nhưng hai batch không được hoán đổi thứ tự và mỗi nhóm mixed chỉ có 4 đến 5 mẫu.
Hai lỗi HTTP 400 ở loại dài / dài và mixed cùng thuộc một câu có input vượt giới hạn `max_model_len` sau khi cộng chat template và chỗ dành cho output; raw record giữ nguyên lỗi đó.
Nhóm mixed có 19 response thành công, chưa đủ để ước lượng p99 ổn định.
Xem [biểu đồ p95 theo loại tải](assets/vllm_composition.svg).

## Capacity, stress và recovery

Giả thuyết: tải cao làm queue và latency tăng trước khi xuất hiện lỗi request hoặc hết VRAM.
Stress dùng 20 request cho concurrency 4, 8 và 16, rồi 32 request cho concurrency 32.
Mỗi request có tối đa 64 output token; sau stress, bốn request tuần tự xác nhận server tiếp tục phục vụ.

| Đồng thời | Thành công | Request/s | Output token/s | TTFT TB | p95 latency | Queue TB | Nhiệt GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 20/20 | 0.141 | 8.78 | 4.05 s | 31.03 s | <0.01 s | 82°C |
| 8 | 20/20 | 0.207 | 12.89 | 8.22 s | 37.37 s | <0.01 s | 80°C |
| 16 | 20/20 | 0.267 | 16.69 | 15.29 s | 53.69 s | 1.99 s | 81°C |
| 32 | 32/32 | 0.427 | 26.46 | 30.88 s | 74.92 s | 11.16 s | 81°C |

Không có timeout hoặc lỗi trong các mức stress đã thử, và lượt recovery đạt 4/4 với latency trung bình 3.27 giây.
Mức 32 có throughput đo được cao nhất nhưng không phù hợp khi cần phản hồi nhanh; mức 4 đến 8 giảm queue rõ rệt nhưng vẫn có p95 latency khoảng 31 đến 37 giây.
Đây là giới hạn thực nghiệm của batch ngắn, không phải bảo đảm cho tải kéo dài nhiều giờ hoặc request context dài.
VRAM peak trong stress khoảng 5034 MiB, bao gồm phần đã được vLLM cấp phát trước.
Xem [biểu đồ stress](assets/vllm_stress.svg).

## Prompt RAG với retrieval TopK 1

Giả thuyết: nhiều câu hỏi trên cùng tài liệu có TTFT thấp hơn nhờ prefix cache, trong khi context dài làm prefill tốn thời gian.
Ba cấu trúc đều dùng 8 request, concurrency 4 và tối đa 64 output token.
Lượt `same-document` hỏi nhiều câu trên cùng một context; `different-documents` dùng context đã retrieval tương ứng từng câu; `long-context` mở rộng context đến khoảng 4500 ký tự.

| Cấu trúc | Input token TB | Cache hit | TTFT TB | Latency TB | Request/s | Thành công |
|---|---:|---:|---:|---:|---:|---:|
| Cùng tài liệu | 481 | 78.5% | 2.27 s | 21.74 s | 0.168 | 8/8 |
| Khác tài liệu | 456 | 24.6% | 5.82 s | 26.60 s | 0.138 | 8/8 |
| Context dài | 1480 | 16.2% | 17.44 s | 45.17 s | 0.088 | 8/8 |

Cùng tài liệu cho TTFT thấp hơn 61.0% so với khác tài liệu và cache hit cao hơn 53.9 điểm phần trăm, dù input token trung bình hơi nhiều hơn.
Lợi ích này áp dụng khi nhiều câu hỏi thực sự chia sẻ cùng tài liệu ở đầu prompt.
Bộ test retrieval một lượt gồm 20 câu với tài liệu khác nhau, nên không thể suy rộng mức giảm này cho mọi truy vấn thực tế.
Biểu đồ [TTFT và cache hit của RAG](assets/vllm_rag.svg) trình bày phép đo.

## Chất lượng câu trả lời (RAGAS)

Trên 100 tài liệu TechQA và 20 câu hỏi, retrieval được chọn tìm đúng gold document ở 20/20 câu, và câu trả lời được chấm đạt faithfulness 0.953 cùng answer correctness 0.943.

| Chỉ số | Giá trị |
|---|---:|
| Hit gold document | 20/20 |
| Faithfulness | 0.953 |
| Answer correctness | 0.943 |
| Answer relevancy | 0.938 |
| Citation precision | 0.960 |
| Context precision | 1.000 |
| Context recall | 1.000 |

Run MLflow `66def2f12b754fee8a70fabea1ef2433` trong experiment `techqa-vllm-ragas` lưu các điểm này, với run gốc `89bb770cf71845d6af1e43a2c4f5bbb3`, model `qwen3.5:4b`, TopK=1, 20 câu hỏi và dataset fingerprint `458ab06a6266426d1fdc0eb6da50984b694a30bce9000835a695a71d844437ca`.

## Cấu hình tối ưu cuối cùng

Giả thuyết: đặt tài liệu trước câu hỏi giúp các câu hỏi tiếp theo trên cùng tài liệu tái dùng prefix KV và giảm TTFT, trong khi không làm chậm câu hỏi đầu tiên.
Đây là phép đo đối chứng cho khuyến nghị rút ra từ nhóm prefix và nhóm RAG ở trên.

Thiết kế dùng 20 tài liệu TechQA, mỗi tài liệu chạy cả hai layout, mỗi layout hỏi hai câu liên tiếp trên cùng context, tổng 80 request với concurrency 1 và tối đa 32 output token.
Thứ tự hai nhánh được đảo theo tài liệu (`AB` cho tài liệu chẵn, `BA` cho tài liệu lẻ) để thứ tự chạy không thiên vị một layout.
Mỗi tài liệu và mỗi nhánh có nonce riêng ở đầu system message, nên không có nhánh nào thừa hưởng prefix của nhánh kia hoặc của scenario trước.
Script [vllm_final_optimization.py](../scripts/vllm_final_optimization.py) kiểm tra số token của từng prompt bằng `/tokenize` trước khi gửi và dừng nếu một request lỗi, nên bảng dưới đây chỉ gồm phép so sánh đủ 80/80 request thành công.

| Layout | Input token TB | TTFT câu đầu | TTFT câu tiếp theo | Latency câu tiếp theo | Cache hit | Thành công |
|---|---:|---:|---:|---:|---:|---:|
| Tài liệu trước (`document_first`) | 391.8 | 2.255 s | 0.834 s | 1.350 s | 41.9% | 40/40 |
| Câu hỏi trước (`question_first`) | 391.4 | 2.253 s | 2.207 s | 2.722 s | 10.0% | 40/40 |

Hai nhánh gần như đồng nhất về chi phí prefill: input token trung bình chênh 0.4 token, chênh lệch token của từng cặp prompt nằm trong khoảng -1 đến +1, và TTFT câu đầu chênh -0.08% (paired t = -0.16 trên 20 cặp, không có khác biệt).
Phép kiểm tra đối chứng này cho thấy khác biệt ở câu tiếp theo đến từ thứ tự token, không phải từ độ dài prompt.

Ở câu hỏi tiếp theo, đặt tài liệu trước giảm TTFT trung bình 62.2% (2.207 s xuống 0.834 s) và giảm latency hoàn tất 50.4%.
Hiệu ứng nhất quán trên từng tài liệu: `document_first` nhanh hơn ở 20/20 cặp, chênh lệch trung bình 1.373 s với paired t = +17.6.
p95 TTFT của câu tiếp theo là 1.389 s so với 2.598 s.
Cache hit tính trên toàn nhánh là 41.9% so với 10.0%, khớp với hướng của nhóm prefix caching.

Khuyến nghị cho cấu hình phục vụ hiện tại là giữ thứ tự `Documents` trước `Question`, giữ phần đầu prompt ổn định giữa các lượt hỏi trên cùng tài liệu, và dùng concurrency thấp khi cần phản hồi tương tác.

Giới hạn của phép đo này cần nêu rõ.
Chỉ 5/40 cặp câu trả lời trùng byte giữa hai layout, nên đổi thứ tự prompt làm thay đổi nội dung sinh ra chứ không chỉ thay đổi tốc độ; phép đo này không đánh giá chất lượng của hai layout.
Lợi ích chỉ xuất hiện khi nhiều câu hỏi thật sự dùng lại cùng tài liệu ở đầu prompt, còn câu hỏi đầu tiên trên một tài liệu mới không nhanh hơn.
Phép đo dùng 32 output token, concurrency 1 và một GPU dùng chung với Ollama, nên không suy rộng cho output dài hoặc tải đồng thời cao.
Run MLflow `20f9bbb608924ce287c3b7e05295ec48` (`document_first`) và `985da1c8c339406daf2211cb82d833a1` (`question_first`) trong experiment `techqa-vllm-inference` giữ cấu hình, metric và raw record của phép so sánh này.
Xem [biểu đồ cấu hình tối ưu](assets/vllm_final_optimization.svg).

## Tái lập và giới hạn

Project dùng Python 3.11 trong virtual environment riêng; dependencies của phép đo ở [requirements/benchmark.txt](../requirements/benchmark.txt).
Giữ nguyên vLLM server hiện có và OpenRouter key trong `.env` khi chạy lại.

```powershell
py -3.11 -m venv .venv-loadtest
& .venv-loadtest/Scripts/python.exe -m pip install -r requirements/benchmark.txt
& .venv-loadtest/Scripts/python.exe -m scripts.vllm_techqa_benchmark --output artifacts/vllm_benchmark_clean
& .venv-loadtest/Scripts/python.exe -m scripts.vllm_final_optimization
& .venv-loadtest/Scripts/python.exe -m scripts.plot_vllm_benchmark
```

Tất cả raw record nằm ở `artifacts/vllm_benchmark_clean/measurements.jsonl` và trong từng MLflow run `measurements/record.json`.
Run MLflow `4a28d0a1827d4eadb483639a9afda711` lưu bảy biểu đồ, danh sách run ID được dùng cho từng scenario và tệp tổng hợp `analysis/plot_summary.json`.
Một lượt cài dependency chạy song song với benchmark và một số kịch bản bị cache warm ngoài ý muốn đã bị loại khỏi phân tích; các scenario liên quan được chạy lại với nonce riêng.
Lỗi, timeout và số request thành công được giữ trong raw record, không thay bằng giá trị ước lượng.
Máy có một GPU dùng chung với Ollama embedding, và `nvidia-smi` chỉ cho thấy tổng VRAM, nên không thể quy toàn bộ mức dùng cho vLLM.
20 câu test đã xuất hiện trong lịch sử lựa chọn retrieval champion của project, nên đây không phải holdout hoàn toàn mới.
