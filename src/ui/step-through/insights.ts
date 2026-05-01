// Frontend-side insights. Currently backend provides insights directly, but this
// lives here so we can augment or override per stage without round-tripping.

import type { Stage } from './types';

export function formatShape(shape: number[] | undefined): string {
  if (!shape) return '—';
  return `[${shape.join(', ')}]`;
}

export function compactShape(shape: number[] | undefined): string {
  if (!shape) return '—';
  // Drop batch dim (first dim = 1 during step-through)
  const inner = shape[0] === 1 ? shape.slice(1) : shape;
  // Matrix notation: [dim1, dim2, ...]
  return `[${inner.join(', ')}]`;
}

/** Fallback insight if backend didn't provide one. */
export function fallbackInsight(stage: Stage): string {
  if (stage.inputShape && stage.outputShape) {
    return `${compactShape(stage.inputShape)} → ${compactShape(stage.outputShape)}`;
  }
  return stage.displayName;
}
