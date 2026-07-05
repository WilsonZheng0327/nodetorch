// Pure helpers for the gradient charts. Kept out of charts.tsx so that file
// exports only components (React Fast Refresh requires component-only modules).

// Per-layer gradient magnitude (RMS = norm / sqrt(params), so layer sizes are
// comparable). `norm` is the L2-norm fallback for older data without rms.
export type GradEntry = { name: string; norm: number; rms?: number };

export const gradMag = (e: GradEntry) => e.rms ?? e.norm;

/**
 * Shared log10 y-domain across the whole run, so scrubbing the depth chart and
 * reading the trend chart both use one fixed absolute scale (no per-epoch
 * renormalization). Returns [loExp, hiExp] as integer powers of 10, or null.
 */
export function gradientLogDomain(
  progress: { gradientFlow?: GradEntry[] }[],
): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const ep of progress) {
    for (const e of ep.gradientFlow ?? []) {
      const m = gradMag(e);
      if (m == null || !isFinite(m) || m <= 0) continue;
      lo = Math.min(lo, m);
      hi = Math.max(hi, m);
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) return null;
  let loExp = Math.floor(Math.log10(lo));
  let hiExp = Math.ceil(Math.log10(hi));
  // Guarantee at least 3 decades so a tight cluster still reads as absolute.
  while (hiExp - loExp < 3) {
    loExp -= 1;
    hiExp += 1;
  }
  return [loExp, hiExp];
}
