// Shared number formatting for all UI views. Previously copy-pasted across
// backprop/*, step-through transformations, and dashboard views with slightly
// different thresholds. Two families live here:
//   - fmt/signed: scientific fallback for values spanning a huge range (gradients,
//     0.3 down to 1e-6) — collapses tiny/huge magnitudes to exponential notation.
//   - fmtValue: adaptive fixed-decimal formatting for activation/feature values,
//     tuned per-view via opts (digits shown shrink as magnitude grows).
//   - fmtAxis: compact axis-tick formatting.

/**
 * Format a value, falling back to scientific notation for tiny/huge magnitudes
 * (|v| < 1e-3 or >= 1e6). Null/undefined render as an em dash.
 */
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

/**
 * Adaptive fixed-decimal formatting: fewer decimals as the magnitude grows, with
 * a scientific fallback below `smallThresh`. Defaults match the pool/norm views;
 * pass opts to reproduce other views (e.g. Conv2d uses `bigDigits: 1`, MSE loss
 * uses finer precision). Preserves each call site's exact prior output.
 */
export function fmtValue(
  v: number,
  opts: {
    bigThresh?: number;
    bigDigits?: number;
    midDigits?: number;
    smallThresh?: number;
    smallDigits?: number;
    expDigits?: number;
  } = {},
): string {
  const {
    bigThresh = 100,
    bigDigits = 0,
    midDigits = 2,
    smallThresh = 0.01,
    smallDigits = 3,
    expDigits = 1,
  } = opts;
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= bigThresh) return v.toFixed(bigDigits);
  if (abs >= 1) return v.toFixed(midDigits);
  if (abs >= smallThresh) return v.toFixed(smallDigits);
  return v.toExponential(expDigits);
}

/** Compact axis-tick formatting. */
export function fmtAxis(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  if (abs >= 0.01) return v.toFixed(2);
  return v.toExponential(1);
}
