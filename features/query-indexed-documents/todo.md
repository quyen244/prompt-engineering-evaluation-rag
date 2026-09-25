# Delivery

- [x] Add Qdrant-backed indexed-document endpoint and bounded query parameters.
- [x] Add indexed-document picker and Top K control to Ask.
- [x] Add a focused frontend contract test; verify backend selection and Top K with a live API query.
- [x] Rebuild/restart affected Compose services and verify a live query.
- [x] Run a reproducible MLflow A/B study: Ollama Qwen3 Embedding vs cached Hugging Face multilingual-e5-base at Top-K 5, 10, and 20.
- [x] Compare Hit Rate, Precision, Recall, MRR, and nDCG on the same 20 source-backed project cases.
- [x] Record the measured comparison in the English README without claiming unmeasured gains.
- [x] Queue PDF ingestion through Celery + Redis and expose status polling.
- [x] Measure three paired synchronous/Celery runs on the real CS229 PDF and log results in MLflow.
- [x] Add the measured ingestion comparison and limitation to the English README.

## 2026-09-25 Study Record

- [x] Run the local embedding x Top-K matrix (Hugging Face multilingual-e5-base and Ollama qwen3-embedding:0.6b; K=5, 10, 20) on the same 20 source-backed smoke cases.
- [x] Verify all six valid variants are recorded in MLflow experiment `rag-embedding-topk-ablation-2026-09-25` (study ID `6e138eddd2c745709942a82c0ac3c170`).
- [x] Publish the measured Hit Rate, Precision, Recall, MRR, and nDCG side by side in the English README; results are retrieval-only and do not justify a serving-default change yet.
- [ ] Run the strongest Hugging Face candidates on the untouched validation split before considering a default change.
