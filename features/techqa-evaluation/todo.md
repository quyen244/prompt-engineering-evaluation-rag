# Delivery

- [x] Replace old `eval_data` inputs with verified copies of the lightweight TechQA files.
- [x] Implement and validate the TechQA dataset adapter and in-memory passage index.
- [x] Run the MLflow-tracked retrieval ablation across 15 configurations and confirm dense Qwen at Top-K 1.
- [x] Run generation evaluation with the approved judge model.
- [x] Run bounded E2E load tests at one and twenty users (scope narrowed from 100; see `docs/inference_optimization.md`).
- [x] Record the Top-K 5 vs Top-K 1 inference comparison and the rejected optimizations in `docs/inference_optimization.md`.
- [ ] Publish measured outcomes and limitations in the README.
- [ ] Measure `OLLAMA_NUM_PARALLEL` 2-4 against the 20-user baseline.
