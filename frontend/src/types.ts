/** Contracts mirror the backend boundary. UI code must not infer evidence or metrics. */
export type ISODate = string;
export type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'stale';
export type Availability = 'available' | 'unavailable' | 'stale';

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'down';
  service: string;
  version?: string;
  checkedAt: ISODate;
}

export interface SourceEvidence {
  sourceId: string;
  documentId: string;
  title: string;
  locator?: string;
  excerpt: string;
  relevance?: number;
  verified: boolean;
}

export interface QueryRequest {
  question: string;
  datasetId?: string;
  promptId?: string;
  modelId?: string;
  runId?: string;
}

export interface QueryResponse {
  queryId: string;
  answer: string;
  evidence: SourceEvidence[];
  datasetId?: string;
  promptId?: string;
  modelId?: string;
  runId?: string;
  traceId?: string;
  createdAt: ISODate;
}

export interface QueryStreamEvent {
  type: 'token' | 'evidence' | 'metadata' | 'done' | 'error';
  value?: string;
  evidence?: SourceEvidence[];
  metadata?: Pick<QueryResponse, 'datasetId' | 'promptId' | 'modelId' | 'runId' | 'traceId'>;
  message?: string;
}

export interface DocumentRecord {
  documentId: string;
  name: string;
  version: string;
  checksum?: string;
  createdAt: ISODate;
  experimentIds: string[];
  status: RunStatus;
  isDemo?: boolean;
}

export interface DocumentHistoryEntry {
  entryId: string;
  documentId: string;
  experimentId: string;
  label: string;
  artifactId?: string;
  createdAt: ISODate;
  status: RunStatus;
  isDemo?: boolean;
}

export interface ExperimentRecord {
  experimentId: string;
  name: string;
  retriever: string;
  reranker?: string;
  datasetId?: string;
  runId?: string;
  status: RunStatus;
  metrics: Record<string, number | null>;
  artifactId?: string;
  createdAt?: ISODate;
  isDemo?: boolean;
}

export interface PromptVersion {
  promptId: string;
  name: string;
  version: string;
  parentPromptId?: string;
  modelId?: string;
  runIds: string[];
  status: 'active' | 'archived' | 'unavailable';
  createdAt?: ISODate;
  isDemo?: boolean;
}

export interface RunRecord {
  runId: string;
  name: string;
  status: RunStatus;
  datasetId?: string;
  promptId?: string;
  modelId?: string;
  traceId?: string;
  startedAt?: ISODate;
  finishedAt?: ISODate;
  error?: string;
  isDemo?: boolean;
}

export interface MonitoringSnapshot {
  availability: Availability;
  checkedAt?: ISODate;
  activeRuns?: number;
  tracesInWindow?: number;
  p95LatencyMs?: number;
  errorRate?: number;
  note?: string;
}

export interface ApiErrorShape {
  code: string;
  message: string;
  retryable: boolean;
}
