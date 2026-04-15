import type { NodeDefinition } from '../../../core/nodedef';

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
        // If only one is connected, pass through its shape
        if (!a) return { outputs: { out: b }, metadata: { outputShape: b } };
        if (!b) return { outputs: { out: a }, metadata: { outputShape: a } };

        // Both connected — shapes must match for element-wise add
        if (JSON.stringify(a) !== JSON.stringify(b)) {
          return {
            outputs: {},
            metadata: { error: `Shape mismatch: [${a.join(', ')}] + [${b.join(', ')}]` },
          };
        }

        return {
          outputs: { out: a },
          metadata: {
            outputShape: a,
            shapes: [{ label: 'Output', value: a }],
          },
        };
      },
    },
  },
};
