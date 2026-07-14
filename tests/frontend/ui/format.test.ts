// Tests for the shared number formatters in src/ui/format.ts. These back several
// visualization views that previously each had their own copy — the fmtValue cases
// pin the exact per-view configurations so the consolidation stays behavior-preserving.

import { describe, it, expect } from 'vitest';
import { fmt, signed, fmtValue, fmtAxis } from '../../../src/ui/format';

describe('fmt — scientific fallback formatter', () => {
  it('renders null/undefined as an em dash', () => {
    expect(fmt(null)).toBe('—');
    expect(fmt(undefined)).toBe('—');
  });

  it('renders exact zero as "0"', () => {
    expect(fmt(0)).toBe('0');
  });

  it('uses fixed decimals in the normal range', () => {
    expect(fmt(0.5)).toBe('0.5000');
    expect(fmt(1.23456, 2)).toBe('1.23');
  });

  it('falls back to scientific notation for tiny magnitudes (< 1e-3)', () => {
    expect(fmt(0.0005)).toBe('5.00e-4');
  });

  it('falls back to scientific notation for huge magnitudes (>= 1e6)', () => {
    expect(fmt(1_234_567)).toBe('1.23e+6');
  });
});

describe('signed — fmt with an explicit leading +', () => {
  it('prefixes non-negative values with +', () => {
    expect(signed(0.5)).toBe('+0.5000');
    expect(signed(0)).toBe('+0');
  });

  it('leaves the natural minus sign on negatives', () => {
    expect(signed(-0.5)).toBe('-0.5000');
  });

  it('still renders null as an em dash (no + prefix)', () => {
    expect(signed(null)).toBe('—');
  });
});

describe('fmtValue — adaptive fixed-decimal formatter', () => {
  it('renders exact zero as "0" regardless of options', () => {
    expect(fmtValue(0)).toBe('0');
    expect(fmtValue(0, { bigDigits: 1 })).toBe('0');
  });

  // Default options match the pool/norm views.
  it('uses the pool/norm defaults: 0 decimals big, 2 mid, 3 small, sci below 0.01', () => {
    expect(fmtValue(150)).toBe('150');
    expect(fmtValue(5.5)).toBe('5.50');
    expect(fmtValue(0.05)).toBe('0.050');
    expect(fmtValue(0.0001)).toBe('1.0e-4');
  });

  it('reproduces the Conv2d view config (bigDigits: 1)', () => {
    expect(fmtValue(150, { bigDigits: 1 })).toBe('150.0');
    // Mid/small tiers unchanged from defaults.
    expect(fmtValue(5.5, { bigDigits: 1 })).toBe('5.50');
  });

  it('reproduces the MSE-loss view config (finer precision, higher thresholds)', () => {
    const mse = {
      bigThresh: 1000,
      bigDigits: 1,
      midDigits: 4,
      smallThresh: 0.0001,
      smallDigits: 6,
      expDigits: 2,
    };
    expect(fmtValue(1500, mse)).toBe('1500.0');
    expect(fmtValue(2.5, mse)).toBe('2.5000');
    expect(fmtValue(0.005, mse)).toBe('0.005000');
    expect(fmtValue(0.00001, mse)).toBe('1.00e-5');
  });

  it('handles negatives by magnitude (sign preserved in output)', () => {
    expect(fmtValue(-5.5)).toBe('-5.50');
    expect(fmtValue(-150)).toBe('-150');
  });
});

describe('fmtAxis — compact axis-tick formatter', () => {
  it('renders zero as "0"', () => {
    expect(fmtAxis(0)).toBe('0');
  });

  it('drops decimals for large ticks', () => {
    expect(fmtAxis(1500)).toBe('1500');
    expect(fmtAxis(150)).toBe('150');
  });

  it('keeps one/two decimals in the mid range', () => {
    expect(fmtAxis(5.5)).toBe('5.5');
    expect(fmtAxis(0.05)).toBe('0.05');
  });

  it('uses scientific notation for very small ticks', () => {
    expect(fmtAxis(0.001)).toBe('1.0e-3');
  });
});
