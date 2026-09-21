import { describe, expect, it } from 'vitest';
import { evidenceLabel, metricLabel } from './uiState';

describe('provenance-safe display state', () => {
  it('labels missing evidence as unavailable instead of fabricating a source', () => {
    expect(evidenceLabel([])).toBe('Unavailable — no verified source evidence returned.');
    expect(evidenceLabel(undefined)).toContain('Unavailable');
  });

  it('labels missing metrics as unavailable', () => {
    expect(metricLabel(null)).toBe('Unavailable');
    expect(metricLabel(undefined)).toBe('Unavailable');
    expect(metricLabel(0)).toBe('0');
  });
});
