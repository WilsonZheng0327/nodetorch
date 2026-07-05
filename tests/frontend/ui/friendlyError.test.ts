// Table test for the PyTorch-error → student-message translator. This copy is
// educational and matched by substring/regex, so it silently degrades if a
// PyTorch version changes its wording — pin the mappings here.

import { describe, it, expect } from 'vitest';
import { friendlyError } from '../../../src/ui/friendlyError';

describe('friendlyError', () => {
  const cases: { raw: string; expect: RegExp }[] = [
    {
      raw: 'RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x784 and 512x10)',
      expect: /Linear layer: input is 64x784 but weights expect 512x10/,
    },
    {
      raw: 'Expected 4-dimensional input for 4-dimensional weight',
      expect: /4D tensor \[B,C,H,W\]/,
    },
    { raw: 'Expected 3-dimensional input', expect: /3D tensor \[B,seq,features\]/ },
    { raw: 'Expected 2-dimensional input', expect: /forget a Flatten layer/ },
    { raw: 'size mismatch, m1: [...], m2: [...]', expect: /Tensor size mismatch/ },
    { raw: 'CUDA out of memory. Tried to allocate ...', expect: /GPU out of memory/ },
    { raw: 'cuda:5 is not a valid device', expect: /device not available/ },
    { raw: 'negative dimension -3 is not allowed', expect: /negative dimension/ },
  ];

  for (const c of cases) {
    it(`translates: ${c.raw.slice(0, 40)}…`, () => {
      expect(friendlyError(c.raw)).toMatch(c.expect);
    });
  }

  it('passes unmatched messages through unchanged', () => {
    const msg = 'Some brand new error PyTorch invented';
    expect(friendlyError(msg)).toBe(msg);
  });

  it('falls through the matmul branch gracefully when shapes are absent', () => {
    // The header matches but there are no NxM shapes to extract → returns raw.
    const msg = 'mat1 and mat2 shapes cannot be multiplied';
    expect(friendlyError(msg)).toBe(msg);
  });
});
