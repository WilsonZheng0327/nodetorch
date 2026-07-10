// Shared number formatting for the backprop views. Gradients span a huge range
// (0.3 down to 1e-6), so tiny/huge magnitudes fall back to scientific notation
// rather than collapsing to "0.0000".

export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined) return '—';
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential(2);
  return v.toFixed(digits);
}

/** Like `fmt`, but always shows a leading + on non-negative values. */
export function signed(v: number | null | undefined, digits = 4): string {
  return v === null || v === undefined ? '—' : (v >= 0 ? '+' : '') + fmt(v, digits);
}
