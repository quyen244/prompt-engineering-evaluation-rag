# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Add durable project-specific notes here as they are discovered through real work.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.

## RAG foundation

- The composable LangChain RAG boundaries live under `src/rag/`; the offline
  contract tests are in `tests/test_rag_core.py` and the architecture overview is
  `docs/rag_core.md`.

## Project notes

- Run the self-contained telemetry contracts with `python3 -m unittest discover -s tests -v`.
- Privacy-safe observability contracts, schema versions, dashboard, alerts, and sink configuration are owned by `docs/telemetry_contracts.md`, `src/telemetry/`, `schemas/telemetry/`, and `infra/`; keep raw request, prompt, answer, credential, and stack-trace content outside all telemetry sinks.
