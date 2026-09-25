import { useCallback, useEffect, useMemo, useState } from 'react';
import { api } from './api';
import type {
  DocumentHistoryEntry,
  DocumentRecord,
  ExperimentRecord,
  HealthResponse,
  IngestJob,
  MonitoringSnapshot,
  PromptVersion,
  QueryResponse,
  RunRecord,
} from './types';
import { evidenceLabel, metricLabel, type LoadState } from './uiState';

 type View = 'ask' | 'documents' | 'experiments' | 'prompts' | 'monitoring';

const viewMeta: Record<View, { label: string; icon: string; hint: string }> = {
  ask: { label: 'Ask', icon: '↗', hint: 'Query your indexed knowledge' },
  documents: { label: 'Documents', icon: '▤', hint: 'Append-only source history' },
  experiments: { label: 'Experiments', icon: '◇', hint: 'Compare retriever runs' },
  prompts: { label: 'Prompt lineage', icon: '⌘', hint: 'Versions, models, and runs' },
  monitoring: { label: 'Monitoring', icon: '◌', hint: 'Health and trace entry point' },
};

function useLoad<T>(loader: () => Promise<T>): [LoadState<T>, () => Promise<void>] {
  const [state, setState] = useState<LoadState<T>>({ status: 'idle' });
  const refresh = useCallback(async () => {
    setState((previous) => ({ status: previous.data ? 'loading' : 'loading', data: previous.data }));
    try {
      const data = await loader();
      setState({ status: 'ready', data });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to load data.';
      setState((previous) => ({ status: previous.data ? 'stale' : 'error', data: previous.data, error: message }));
    }
  }, [loader]);
  return [state, refresh];
}

function StatusBadge({ status }: { status: string }) {
  return <span className={`status status-${status}`}>{status}</span>;
}

function Unavailable({ label = 'Unavailable' }: { label?: string }) {
  return <span className="unavailable"><span aria-hidden="true">—</span> {label}</span>;
}

function LoadNotice({ state, onRetry }: { state: LoadState<unknown>; onRetry: () => void }) {
  if (state.status === 'loading') return <div className="notice notice-loading"><span className="spinner" /> Loading live data…</div>;
  if (state.status === 'error') return <div className="notice notice-error"><strong>Live data unavailable.</strong> {state.error}<button className="text-button" onClick={onRetry}>Retry</button></div>;
  if (state.status === 'stale') return <div className="notice notice-stale"><strong>Showing stale data.</strong> {state.error}<button className="text-button" onClick={onRetry}>Refresh</button></div>;
  return null;
}

function IdentityStrip({ response }: { response?: QueryResponse }) {
  const identities = [
    ['Documents', response?.documentIds?.join(', ') || response?.datasetId],
    ['Prompt', response?.promptId],
    ['Model', response?.modelId],
    ['Run', response?.runId],
    ['Trace', response?.traceId],
  ];
  return <div className="identity-strip" aria-label="Run identity">
    {identities.map(([label, value]) => <div className="identity" key={label}><span>{label}</span><strong>{value || 'Unavailable'}</strong></div>)}
  </div>;
}

function AskView() {
  const documentsLoader = useCallback(() => api.listIndexedDocuments(), []);
  const [documentsState, retryDocuments] = useLoad<DocumentRecord[]>(documentsLoader);
  useEffect(() => { void retryDocuments(); }, [retryDocuments]);
  const [question, setQuestion] = useState('');
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[] | null>(null);
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<LoadState<QueryResponse>>({ status: 'idle' });
  const [lastQuestion, setLastQuestion] = useState('');
  const documents = documentsState.status === 'ready' || documentsState.status === 'stale' ? documentsState.data ?? [] : [];
  const selectedIds = selectedDocumentIds ?? documents.map((document) => document.documentId);
  const toggleDocument = (documentId: string) => {
    const current = selectedDocumentIds ?? documents.map((document) => document.documentId);
    const next = current.includes(documentId) ? current.filter((id) => id !== documentId) : [...current, documentId];
    setSelectedDocumentIds(next.length === documents.length ? null : next);
  };
  const ask = async () => {
    const trimmed = question.trim();
    if (!trimmed || selectedIds.length === 0) return;
    setLastQuestion(trimmed);
    setResult({ status: 'loading' });
    try {
      setResult({ status: 'ready', data: await api.query({ question: trimmed, documentIds: selectedIds, topK }) });
    } catch (error) {
      setResult({ status: 'error', error: error instanceof Error ? error.message : 'Query failed.', data: undefined });
    }
  };
  const answer = result.status === 'ready' ? result.data : undefined;
  return <section className="view ask-view" aria-labelledby="ask-title">
    <div className="eyebrow">RAG workspace <span className="dot" /> provenance first</div>
    <div className="view-heading"><div><h1 id="ask-title">Ask your knowledge base</h1><p>Answers stay tied to the exact dataset, prompt, model, run, and trace that produced them.</p></div><span className="live-pill"><span className="live-dot" /> API connection</span></div>
    <div className="ask-grid">
      <div className="ask-main">
        <div className="question-card">
          <div className="query-controls">
            <div className="document-picker">
              <div className="query-control-heading"><label>Indexed documents</label><div className="selection-actions"><button type="button" className="text-button" onClick={() => setSelectedDocumentIds(null)} disabled={!documents.length}>Select all</button><button type="button" className="text-button" onClick={() => setSelectedDocumentIds([])} disabled={!documents.length}>Clear</button></div></div>
              <LoadNotice state={documentsState} onRetry={() => void retryDocuments()} />
              {documents.length ? <div className="document-options">{documents.map((document) => <label className="document-option" key={document.documentId}><input type="checkbox" checked={selectedIds.includes(document.documentId)} onChange={() => toggleDocument(document.documentId)} /><span><strong>{document.name}</strong><small>{document.documentId}</small></span></label>)}</div> : documentsState.status !== 'loading' && <p className="control-empty">No indexed documents found. Upload a PDF from Documents first.</p>}
            </div>
            <div className="top-k-control"><label htmlFor="top-k">Passages to retrieve</label><input id="top-k" type="number" min={1} max={50} value={topK} onChange={(event) => setTopK(Math.max(1, Math.min(50, Number(event.target.value) || 1)))} /><small>Top K, 1-50</small></div>
          </div>
          <label htmlFor="question">Question</label>
          <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask a question about your indexed documents…" rows={4} onKeyDown={(event) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void ask(); }} />
          <div className="composer-footer"><span className="keyboard-hint">{selectedIds.length} document{selectedIds.length === 1 ? '' : 's'} · {topK} passages</span><button className="primary-button" onClick={() => void ask()} disabled={!question.trim() || selectedIds.length === 0 || result.status === 'loading' || documentsState.status === 'loading'}>{result.status === 'loading' ? 'Running…' : 'Run query'} <span aria-hidden="true">↗</span></button></div>
        </div>
        {result.status === 'error' && <div className="notice notice-error query-error"><strong>Could not run query.</strong> {result.error}<button className="text-button" onClick={() => void ask()}>Try again</button></div>}
        {result.status === 'loading' && <div className="answer-card loading-card"><div className="skeleton skeleton-wide" /><div className="skeleton skeleton-medium" /><div className="skeleton skeleton-short" /></div>}
        {answer && <div className="answer-card"><div className="answer-header"><div><span className="card-kicker">Answer</span><h2>{lastQuestion}</h2></div><StatusBadge status="completed" /></div><p className="answer-copy">{answer.answer || <Unavailable label="No answer returned" />}</p><IdentityStrip response={answer} /><div className="evidence-heading"><h3>Source evidence</h3><span>{evidenceLabel(answer.evidence)}</span></div>{answer.evidence.length ? <div className="source-list">{answer.evidence.map((source) => <article className="source-card" key={source.sourceId}><div className="source-top"><span className="source-id">{source.sourceId}</span>{source.verified ? <span className="verified">✓ verified</span> : <span className="unavailable">unverified</span>}</div><h4>{source.title}</h4><p>{source.excerpt}</p>{source.locator && <small>{source.locator}</small>}</article>)}</div> : <div className="empty-evidence"><span className="empty-icon">⌕</span><strong>Evidence unavailable</strong><p>No verified source citations were returned for this answer. This console will not infer or fabricate a citation.</p></div>}</div>}
        {result.status === 'idle' && <div className="empty-answer"><div className="empty-icon large">✦</div><h2>Ask a question to start</h2><p>Your response and its evidence will appear here. No connected data is shown as measured.</p></div>}
      </div>
      <aside className="ask-side"><div className="side-card"><div className="card-kicker">Query contract</div><h3>Trace every answer</h3><p>Each response should expose the identities needed to reproduce it. Missing fields remain visibly unavailable.</p><div className="contract-list">{['Selected documents', 'Prompt version', 'Model identity', 'Run identity', 'Trace identity'].map((item) => <div key={item}><span>{item}</span><Unavailable /></div>)}</div></div></aside>
    </div>
  </section>;
}

function DocumentsView() {
  const loader = useCallback(() => api.listDocuments(), []);
  const [state, retry] = useLoad<DocumentRecord[]>(loader);
  const historyLoader = useCallback(() => api.documentHistory(), []);
  const [history, retryHistory] = useLoad<DocumentHistoryEntry[]>(historyLoader);
  const [showForm, setShowForm] = useState(false);
  const [docName, setDocName] = useState('');
  const [docContent, setDocContent] = useState('');
  const [docFile, setDocFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState('');

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!docFile && (!docName.trim() || !docContent.trim())) return;
    setSubmitting(true);
    try {
      if (docFile) {
        let job: IngestJob = await api.uploadDocument(docFile);
        setIngestStatus('Queued for background ingestion');
        const ingestionDeadline = Date.now() + 30 * 60 * 1000;
        while (job.status === 'queued' || job.status === 'started') {
          if (Date.now() >= ingestionDeadline) throw new Error('PDF ingestion timed out while waiting for the worker.');
          await new Promise((resolve) => window.setTimeout(resolve, 1000));
          job = await api.ingestJob(job.taskId);
          setIngestStatus(job.status === 'started' ? 'Extracting, embedding, and indexing PDF' : 'Waiting for worker');
        }
        if (job.status !== 'completed') throw new Error(job.error || 'PDF ingestion failed.');
      } else {
        await api.ingestDocument({ name: docName.trim(), content: docContent.trim() });
      }
      setDocName('');
      setDocContent('');
      setDocFile(null);
      setIngestStatus('');
      setShowForm(false);
      void retry();
    } catch (err) {
      alert('Ingestion failed: ' + String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const documents = state.status === 'ready' || state.status === 'stale' ? state.data ?? [] : [];
  const entries = history.status === 'ready' || history.status === 'stale' ? history.data ?? [] : [];
  return (
    <section className="view" aria-labelledby="documents-title">
      <div className="view-heading">
        <div>
          <div className="eyebrow">Corpus / immutable trail</div>
          <h1 id="documents-title">Document history</h1>
          <p>Track every ingestion artifact without overwriting the source trail.</p>
        </div>
        <button className="secondary-button" onClick={() => setShowForm(!showForm)}>
          {showForm ? '✕ Close form' : '＋ Ingest document'}
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleIngest} style={{ background: '#111827', padding: '1.25rem', borderRadius: '8px', border: '1px solid #374151', marginBottom: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <h3 style={{ margin: 0, fontSize: '1rem', color: '#60a5fa' }}>Ingest New Knowledge Document</h3>
          <label style={{ display: 'grid', gap: '0.4rem', color: '#d1d5db' }}>
            Upload a PDF for local indexing
            <input type="file" accept="application/pdf,.pdf" onChange={(e) => setDocFile(e.target.files?.[0] ?? null)} />
          </label>
          <div style={{ color: '#9ca3af', fontSize: '0.8rem' }}>Or paste text below. PDF uploads are embedded by the configured Ollama model.</div>
          <input
            type="text"
            placeholder="Document title (e.g. system_architecture_spec.md)"
            value={docName}
            onChange={(e) => setDocName(e.target.value)}
            required={!docFile}
            style={{ background: '#1f2937', color: '#f3f4f6', border: '1px solid #4b5563', padding: '0.5rem 0.75rem', borderRadius: '6px' }}
          />
          <textarea
            rows={5}
            placeholder="Paste document text or markdown content here..."
            value={docContent}
            onChange={(e) => setDocContent(e.target.value)}
            required={!docFile}
            style={{ background: '#1f2937', color: '#f3f4f6', border: '1px solid #4b5563', padding: '0.5rem 0.75rem', borderRadius: '6px', fontFamily: 'monospace', fontSize: '0.85rem' }}
          />
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button type="submit" className="primary-button" disabled={submitting}>
              {submitting ? 'Ingesting...' : 'Ingest into Knowledge Base'}
            </button>
            <button type="button" className="secondary-button" onClick={() => setShowForm(false)}>
              Cancel
            </button>
          </div>
          {ingestStatus && <div role="status" aria-live="polite" style={{ color: '#9ca3af', fontSize: '0.85rem' }}>{ingestStatus}</div>}
        </form>
      )}

      <LoadNotice state={state} onRetry={() => void retry()} />
      {documents.length === 0 && state.status !== 'loading' ? (
        <EmptyData title="No documents connected" copy="The document index has not returned any records. Connect the API to inspect real sources." />
      ) : (
        <div className="table-card">
          <table>
            <thead>
              <tr>
                <th>Document</th>
                <th>Version</th>
                <th>Status</th>
                <th>Experiments</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.documentId}>
                  <td><strong>{doc.name}</strong><small>{doc.documentId}{doc.isDemo && <em>Demo fixture</em>}</small></td>
                  <td>{doc.version}</td>
                  <td><StatusBadge status={doc.status} /></td>
                  <td>{doc.experimentIds.length || <Unavailable />}</td>
                  <td>{doc.createdAt || <Unavailable />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="section-heading">
        <div>
          <div className="card-kicker">Append-only artifacts</div>
          <h2>Experiment trail</h2>
        </div>
        <span className="subtle">Newest first · never inferred</span>
      </div>
      <LoadNotice state={history} onRetry={() => void retryHistory()} />
      {entries.length === 0 && history.status !== 'loading' ? (
        <EmptyData compact title="No history entries" copy="Artifacts will appear here when the backend records a document experiment." />
      ) : (
        <div className="timeline">
          {entries.map((entry) => (
            <div className="timeline-item" key={entry.entryId}>
              <div className="timeline-marker" />
              <div className="timeline-content">
                <div className="timeline-title"><strong>{entry.label}</strong><StatusBadge status={entry.status} /></div>
                <p>{entry.documentId} <span>→</span> {entry.artifactId || 'Result artifact unavailable'}</p>
                <small>{entry.createdAt || 'Timestamp unavailable'} {entry.isDemo && <em>Demo fixture · not measured</em>}</small>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ExperimentsView() {
  const loader = useCallback(() => api.listExperiments(), []);
  const [state, retry] = useLoad<ExperimentRecord[]>(loader);
  const experiments = state.status === 'ready' || state.status === 'stale' ? state.data ?? [] : [];
  return <section className="view" aria-labelledby="experiments-title"><div className="view-heading"><div><div className="eyebrow">Evaluation / comparison</div><h1 id="experiments-title">Retriever experiments</h1><p>Compare retrieval configurations only when measured artifacts are available.</p></div><button className="secondary-button" disabled>＋ New experiment</button></div><LoadNotice state={state} onRetry={() => void retry()} /><div className="compare-toolbar"><span>Compare selected runs</span><span className="subtle">Metrics require a dataset and completed run</span></div>{experiments.length === 0 && state.status !== 'loading' ? <div className="comparison-empty"><div className="empty-icon large">◇</div><h2>No measured experiments</h2><p>Connect an evaluation API to compare BM25-only with BM25-plus-reranker or other retriever runs. Quality numbers will stay <strong>Unavailable</strong> until measured.</p><div className="comparison-ghost"><div><strong>BM25-only</strong><span>Result artifact unavailable</span></div><span className="arrow">→</span><div><strong>BM25 + reranker</strong><span>Result artifact unavailable</span></div></div></div> : <div className="experiment-grid">{experiments.map((experiment) => <ExperimentCard key={experiment.experimentId} experiment={experiment} />)}</div>}</section>;
}

function ExperimentCard({ experiment }: { experiment: ExperimentRecord }) {
  return <article className="experiment-card"><div className="experiment-top"><span className="selection-box" /> <StatusBadge status={experiment.status} /></div><h2>{experiment.name}{experiment.isDemo && <em>Demo fixture · not measured</em>}</h2><p className="retriever-line">{experiment.retriever}{experiment.reranker && <><span>＋</span>{experiment.reranker}</>}</p><dl className="metric-list">{['Hit rate @5', 'NDCG @5', 'Faithfulness'].map((metric) => <div key={metric}><dt>{metric}</dt><dd>{metricLabel(experiment.metrics[metric])}</dd></div>)}</dl><footer><span>Dataset: {experiment.datasetId || 'Unavailable'}</span><span>Run: {experiment.runId || 'Unavailable'}</span></footer></article>;
}

function PromptsView() {
  const promptLoader = useCallback(() => api.listPrompts(), []);
  const [prompts, retry] = useLoad<PromptVersion[]>(promptLoader);
  const runLoader = useCallback(() => api.listRuns(), []);
  const [runs, retryRuns] = useLoad<RunRecord[]>(runLoader);
  const promptRecords = prompts.status === 'ready' || prompts.status === 'stale' ? prompts.data ?? [] : [];
  const runRecords = runs.status === 'ready' || runs.status === 'stale' ? runs.data ?? [] : [];
  return <section className="view" aria-labelledby="prompts-title"><div className="view-heading"><div><div className="eyebrow">Registry / lineage</div><h1 id="prompts-title">Prompt lineage</h1><p>See which prompt version and model travelled through each run.</p></div><button className="secondary-button" disabled>Open registry ↗</button></div><LoadNotice state={prompts} onRetry={() => void retry()} />{promptRecords.length === 0 && prompts.status !== 'loading' ? <EmptyData title="No prompt lineage connected" copy="Prompt versions and parent relationships will be rendered here when the registry API is available." /> : <div className="lineage-list">{promptRecords.map((prompt) => <article className="lineage-card" key={prompt.promptId}><div className="lineage-node"><span className="node-icon">⌘</span><div><strong>{prompt.name}{prompt.isDemo && <em>Demo fixture · not measured</em>}</strong><span>{prompt.promptId} · v{prompt.version}</span></div></div><div className="lineage-connector">↓</div><div className="lineage-meta"><div><span>Parent</span><strong>{prompt.parentPromptId || 'Unavailable'}</strong></div><div><span>Model</span><strong>{prompt.modelId || 'Unavailable'}</strong></div><div><span>Runs</span><strong>{prompt.runIds.length || 'Unavailable'}</strong></div><StatusBadge status={prompt.status} /></div></article>)}</div>}<div className="section-heading"><div><div className="card-kicker">Execution records</div><h2>Runs using these prompts</h2></div></div><LoadNotice state={runs} onRetry={() => void retryRuns()} />{runRecords.length === 0 && runs.status !== 'loading' ? <EmptyData compact title="No runs connected" copy="Run and trace identities are unavailable until the runs API responds." /> : <div className="run-table">{runRecords.map((run) => <div className="run-row" key={run.runId}><span className="run-symbol">●</span><div><strong>{run.name}{run.isDemo && <em>Demo fixture · not measured</em>}</strong><small>{run.runId}</small></div><StatusBadge status={run.status} /><span>{run.modelId || 'Model unavailable'}</span><span>{run.traceId || 'Trace unavailable'}</span></div>)}</div>}</section>;
}

function MonitoringView() {
  const loader = useCallback(() => api.monitoring(), []);
  const [state, retry] = useLoad<MonitoringSnapshot>(loader);
  const snapshot = state.status === 'ready' || state.status === 'stale' ? state.data : undefined;
  return <section className="view" aria-labelledby="monitoring-title"><div className="view-heading"><div><div className="eyebrow">Operations / trace entry</div><h1 id="monitoring-title">Monitoring</h1><p>A read-only entry point for health, runs, metrics, and traces.</p></div><button className="secondary-button" onClick={() => void retry()}>↻ Refresh</button></div><LoadNotice state={state} onRetry={() => void retry()} /><div className="monitor-grid"><article className="monitor-hero"><div className="card-kicker">Service health</div><div className="health-row"><span className={`health-orb ${snapshot?.availability === 'available' ? 'healthy' : ''}`} /><div><h2>{snapshot?.availability === 'available' ? 'Connected' : 'Health unavailable'}</h2><p>{snapshot?.note || 'The monitoring API has not returned a snapshot.'}</p></div></div><div className="monitor-meta"><span>Checked at</span><strong>{snapshot?.checkedAt || 'Unavailable'}</strong></div></article>{[['Active runs', snapshot?.activeRuns], ['Traces / window', snapshot?.tracesInWindow], ['p95 latency', snapshot?.p95LatencyMs ? `${snapshot.p95LatencyMs} ms` : undefined], ['Error rate', snapshot?.errorRate ? `${snapshot.errorRate}%` : undefined]].map(([label, value]) => <article className="stat-card" key={String(label)}><span>{label}</span><strong>{value ?? 'Unavailable'}</strong><small>{value === undefined ? 'No measured value' : 'Reported by API'}</small></article>)}</div><div className="trace-entry"><div><div className="card-kicker">Trace explorer</div><h2>Open traces when a run is selected</h2><p>Trace links are generated from backend identity. This shell never invents telemetry.</p></div><button className="secondary-button" disabled>Open trace explorer ↗</button></div></section>;
}

function EmptyData({ title, copy, compact = false }: { title: string; copy: string; compact?: boolean }) {
  return <div className={`empty-data ${compact ? 'compact' : ''}`}><span className="empty-icon">○</span><div><strong>{title}</strong><p>{copy}</p></div></div>;
}

export default function App() {
  const [view, setView] = useState<View>('ask');
  const healthLoader = useCallback(() => api.health(), []);
  const [health, retryHealth] = useLoad<HealthResponse>(healthLoader);
  useEffect(() => { void retryHealth(); }, [retryHealth]);
  const healthStatus = health.status === 'ready' ? health.data.status : 'unavailable';
  const content = useMemo(() => ({ ask: <AskView />, documents: <DocumentsView />, experiments: <ExperimentsView />, prompts: <PromptsView />, monitoring: <MonitoringView /> }[view]), [view]);
  return <div className="app-shell"><aside className="sidebar"><div className="brand"><div className="brand-mark">✦</div><div><strong>RAG Lab</strong><span>console</span></div></div><div className="workspace-switcher"><span className="workspace-icon">⌁</span><div><small>Workspace</small><strong>provenance-lab</strong></div><span className="chevron">⌄</span></div><nav aria-label="Primary navigation">{(Object.keys(viewMeta) as View[]).map((key) => <button key={key} className={`nav-item ${view === key ? 'active' : ''}`} onClick={() => setView(key)} aria-current={view === key ? 'page' : undefined}><span className="nav-icon" aria-hidden="true">{viewMeta[key].icon}</span><span>{viewMeta[key].label}</span>{key === 'monitoring' && <span className="nav-dot" />}</button>)}</nav><div className="sidebar-bottom"><div className="connection-card"><div className="connection-top"><span className={`connection-dot ${healthStatus === 'ok' ? 'ok' : ''}`} /><span>API status</span><button aria-label="Refresh API status" onClick={() => void retryHealth()}>↻</button></div><strong>{healthStatus === 'ok' ? 'Connected' : 'Unavailable'}</strong><small>{health.status === 'ready' ? health.data.service : 'No live response'}</small></div><div className="user-row"><span className="avatar">QL</span><div><strong>Reviewer</strong><small>Read-only access</small></div><span>•••</span></div></div></aside><main className="main-content"><header className="topbar"><div className="breadcrumb"><span>Workspace</span><span>/</span><strong>{viewMeta[view].label}</strong></div><div className="top-actions"><span className="environment"><span className="env-dot" /> Development</span><button className="icon-button" aria-label="Help">?</button><button className="icon-button" aria-label="Notifications">♧</button></div></header>{content}<footer className="app-footer"><span>RAG Lab Console · provenance-first shell</span><span>Data is shown only when returned by the API</span></footer></main></div>;
}
