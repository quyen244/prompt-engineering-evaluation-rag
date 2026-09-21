# RAG Lab Console

A React/TypeScript frontend shell for the provenance-first RAG reviewer workflow.

```bash
npm install
npm run dev
npm test
npm run build
```

The UI calls `/api` and intentionally has no fixture fallback. Missing evidence,
metrics, identities, traces, and timestamps render as **Unavailable**. Backend
responses can mark individual records with `isDemo`; the UI labels those records
as demo fixtures and does not treat them as measurements.

Typed API boundaries live in `src/types.ts` and `src/api.ts` for health, query
and SSE streaming, documents/history, experiments, prompts, runs, and monitoring.
