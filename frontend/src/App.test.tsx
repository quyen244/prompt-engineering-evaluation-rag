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
});
