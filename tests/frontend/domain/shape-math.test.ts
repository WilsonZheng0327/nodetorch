import { describe, it, expect } from 'vitest';
import { convOutputSize, convTransposeOutputSize } from '../../../src/domain/shape-math';

describe('convOutputSize', () => {
  it('3x3 conv, no padding, stride 1 shrinks by kernel-1', () => {
    expect(convOutputSize(28, 3, 0, 1)).toBe(26);
  });
  it('same padding (pad = (k-1)/2) preserves size', () => {
    expect(convOutputSize(28, 3, 1, 1)).toBe(28);
  });
  it('2x2 pool, stride 2 halves', () => {
    expect(convOutputSize(28, 2, 0, 2)).toBe(14);
  });
  it('floors on non-divisible strides', () => {
    expect(convOutputSize(7, 2, 0, 2)).toBe(3); // floor((7-2)/2)+1 = 3
  });
});

describe('convTransposeOutputSize', () => {
  it('stride 2 doubles (inverse of a stride-2 pool)', () => {
    expect(convTransposeOutputSize(14, 2, 0, 2, 0)).toBe(28);
  });
  it('accounts for output padding', () => {
    expect(convTransposeOutputSize(14, 3, 1, 2, 1)).toBe((14 - 1) * 2 - 2 * 1 + 3 + 1);
  });
});
