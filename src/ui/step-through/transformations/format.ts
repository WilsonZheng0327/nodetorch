// Pure formatting helpers for the step-through transformation views. Kept out of
// shared.tsx (which exports components) so both files stay Fast-Refresh-friendly
// — see react-refresh/only-export-components.

/** Adaptive axis formatting. */
export function fmtAxis(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  if (abs >= 0.01) return v.toFixed(2);
  return v.toExponential(1);
}
