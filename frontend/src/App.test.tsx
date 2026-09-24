import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('API unavailable'))));
});
afterEach(() => cleanup());

describe('RAG Lab console', () => {
  it('does not show fabricated evaluation values when the API is unavailable', async () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'Experiments' }));
    expect(await screen.findByText('No measured experiments')).toBeTruthy();
    expect(screen.queryByText('0.92')).toBeNull();
  });

  it('exposes the main provenance views', () => {
    render(<App />);
    expect(screen.getByRole('button', { name: 'Ask' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Documents' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Experiments' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Prompt lineage' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Monitoring' })).toBeTruthy();
  });

  it('queries the selected indexed documents with the chosen Top K', async () => {
    const queryBody = vi.fn();
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/health')) return Promise.resolve(Response.json({ status: 'ok', service: 'rag', checkedAt: 'now' }));
      if (url.endsWith('/api/documents/indexed')) return Promise.resolve(Response.json([
        { documentId: 'doc-a', name: 'Probability.pdf', version: 'v1', createdAt: 'now', experimentIds: [], status: 'completed' },
        { documentId: 'doc-b', name: 'Statistics.pdf', version: 'v1', createdAt: 'now', experimentIds: [], status: 'completed' },
      ]));
      if (url.endsWith('/api/query')) {
        queryBody(JSON.parse(String(init?.body)));
        return Promise.resolve(Response.json({ queryId: 'q1', answer: 'Bayes theorem answer', evidence: [], createdAt: 'now', documentIds: ['doc-a'], modelId: 'qwen' }));
      }
      return Promise.reject(new Error(`Unexpected request: ${url}`));
    }));

    render(<App />);
    const probability = await screen.findByRole('checkbox', { name: /Probability\.pdf/ });
    const statistics = screen.getByRole('checkbox', { name: /Statistics\.pdf/ });
    fireEvent.click(statistics);
    fireEvent.change(screen.getByLabelText('Passages to retrieve'), { target: { value: '3' } });
    fireEvent.change(screen.getByLabelText('Question'), { target: { value: 'What is Bayes theorem?' } });
    fireEvent.click(screen.getByRole('button', { name: /Run query/ }));

    expect(await screen.findByText('Bayes theorem answer')).toBeTruthy();
    expect((probability as HTMLInputElement).checked).toBe(true);
    expect((statistics as HTMLInputElement).checked).toBe(false);
    expect(queryBody).toHaveBeenCalledWith({ question: 'What is Bayes theorem?', documentIds: ['doc-a'], topK: 3 });
  });
});
