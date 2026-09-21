import type {
  DocumentHistoryEntry,
  DocumentRecord,
  ExperimentRecord,
  HealthResponse,
  MonitoringSnapshot,
  PromptVersion,
  QueryRequest,
  QueryResponse,
  QueryStreamEvent,
  RunRecord,
} from './types';

export class ApiUnavailableError extends Error {
  readonly code = 'API_UNAVAILABLE';
  constructor(message = 'The RAG API is not connected.') {
    super(message);
    this.name = 'ApiUnavailableError';
  }
}

export interface RagApi {
  health(): Promise<HealthResponse>;
  query(request: QueryRequest): Promise<QueryResponse>;
  streamQuery(request: QueryRequest): AsyncGenerator<QueryStreamEvent>;
  listDocuments(): Promise<DocumentRecord[]>;
  documentHistory(documentId?: string): Promise<DocumentHistoryEntry[]>;
  listExperiments(): Promise<ExperimentRecord[]>;
  listPrompts(): Promise<PromptVersion[]>;
  listRuns(): Promise<RunRecord[]>;
  monitoring(): Promise<MonitoringSnapshot>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  });
  if (!response.ok) {
    throw new ApiUnavailableError(`API request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}

async function* streamQuery(requestBody: QueryRequest): AsyncGenerator<QueryStreamEvent> {
  const response = await fetch('/api/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(requestBody),
  });
  if (!response.ok || !response.body) throw new ApiUnavailableError(`Streaming API request failed (${response.status}).`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';
      for (const event of events) {
        const payload = event.split('\n').find((line) => line.startsWith('data:'))?.slice(5).trim();
        if (payload) yield JSON.parse(payload) as QueryStreamEvent;
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}

/** Real API client. It intentionally has no local fixture fallback. */
export const api: RagApi = {
  health: () => request<HealthResponse>('/health'),
  query: (body) => request<QueryResponse>('/query', { method: 'POST', body: JSON.stringify(body) }),
  streamQuery: (body) => streamQuery(body),
  listDocuments: () => request<DocumentRecord[]>('/documents'),
  documentHistory: (documentId) => request<DocumentHistoryEntry[]>(`/documents/history${documentId ? `?documentId=${encodeURIComponent(documentId)}` : ''}`),
  listExperiments: () => request<ExperimentRecord[]>('/experiments'),
  listPrompts: () => request<PromptVersion[]>('/prompts'),
  listRuns: () => request<RunRecord[]>('/runs'),
  monitoring: () => request<MonitoringSnapshot>('/monitoring'),
};
