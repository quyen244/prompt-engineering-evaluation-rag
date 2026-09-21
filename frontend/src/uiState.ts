import type { SourceEvidence } from './types';

export type LoadState<T> =
  | { status: 'idle' | 'loading'; data?: T; error?: undefined }
  | { status: 'ready'; data: T; error?: undefined }
  | { status: 'error' | 'stale'; data?: T; error: string };

export function evidenceLabel(evidence: SourceEvidence[] | undefined): string {
  if (!evidence || evidence.length === 0) return 'Unavailable — no verified source evidence returned.';
  return `${evidence.length} verified source${evidence.length === 1 ? '' : 's'} returned`;
}

export function metricLabel(value: number | null | undefined): string {
  return typeof value === 'number' ? String(value) : 'Unavailable';
}

export function isUnavailable<T>(state: LoadState<T>): boolean {
  return state.status === 'error' || state.status === 'stale' || (state.status === 'ready' && state.data === undefined);
}
