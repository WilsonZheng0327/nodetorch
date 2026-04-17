import type { NodeDefinition } from '../../../core/nodedef';

/** Check if two shapes are broadcast-compatible (PyTorch rules).
 *  Returns the broadcast output shape, or null if incompatible. */
function broadcastShapes(a: number[], b: number[]): number[] | null {
  const rank = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < rank; i++) {
    const da = i < a.length ? a[a.length - 1 - i] : 1;
    const db = i < b.length ? b[b.length - 1 - i] : 1;
    if (da !== db && da !== 1 && db !== 1) return null;
    out.unshift(Math.max(da, db));
  }
  return out;
}

export const addNode: NodeDefinition = {
  type: 'ml.structural.add',
  version: 1,
  displayName: 'Add',
  description: 'Element-wise addition (for skip/residual connections)',
  category: ['ML', 'Structural'],

  getProperties: () => [],

  getPorts: () => [
    { id: 'a', name: 'A', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'b', name: 'B', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const a = inputs.a;
        const b = inputs.b;

        if (!a && !b) return { outputs: {} };
        if (!a) return { outputs: { out: b }, metadata: { outputShape: b } };
        if (!b) return { outputs: { out: a }, metadata: { outputShape: a } };

        const result = broadcastShapes(a, b);
        if (!result) {
          return {
            outputs: {},
            metadata: { error: `Shapes not broadcast-compatible: [${a.join(', ')}] + [${b.join(', ')}]` },
          };
        }

        return {
          outputs: { out: result },
          metadata: {
            outputShape: result,
            shapes: [{ label: 'Output', value: result }],
          },
        };
      },
    },
  },
};
