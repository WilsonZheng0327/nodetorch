// Concat — concatenates N tensors along a specified dimension.
// Dynamic ports: numInputs property controls how many input ports exist.

import type { NodeDefinition } from '../../../core/nodedef';

export const concatNode: NodeDefinition = {
  type: 'ml.structural.concat',
  version: 1,
  displayName: 'Concat',
  description: 'Concatenate tensors along a dimension',
  category: ['ML', 'Structural'],

  getProperties: () => [
    {
      id: 'numInputs',
      name: 'Number of Inputs',
      type: { kind: 'number', min: 2, max: 8, integer: true },
      defaultValue: 2,
      affects: 'ports',
    },
    {
      id: 'dim',
      name: 'Dimension',
      type: { kind: 'number', min: 0, integer: true },
      defaultValue: 1,
      affects: 'execution',
    },
  ],

  getPorts: (properties) => {
    const n = properties.numInputs ?? 2;
    return [
      ...Array.from({ length: n }, (_, i) => ({
        id: `in_${i}`,
        name: `Input ${i}`,
        direction: 'input' as const,
        dataType: 'tensor',
        allowMultiple: false,
        optional: i >= 2, // first two required, rest optional
      })),
      { id: 'out', name: 'Output', direction: 'output' as const, dataType: 'tensor', allowMultiple: true, optional: false },
    ];
  },

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const dim = properties.dim ?? 1;
        const shapes: number[][] = [];

        for (const [key, val] of Object.entries(inputs)) {
          if (key.startsWith('in_') && val) shapes.push(val);
        }

        if (shapes.length === 0) return { outputs: {} };

        // Validate: all shapes must match except along concat dim
        const first = shapes[0];
        for (let i = 1; i < shapes.length; i++) {
          if (shapes[i].length !== first.length) {
            return { outputs: {}, metadata: { error: `Rank mismatch: [${first}] vs [${shapes[i]}]` } };
          }
          for (let d = 0; d < first.length; d++) {
            if (d !== dim && shapes[i][d] !== first[d]) {
              return { outputs: {}, metadata: { error: `Shape mismatch at dim ${d}: ${first[d]} vs ${shapes[i][d]}` } };
            }
          }
        }

        // Output shape: sum along concat dim
        const outShape = [...first];
        outShape[dim] = shapes.reduce((sum, s) => sum + s[dim], 0);

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            shapes: [{ label: 'Output', value: outShape }],
          },
        };
      },
    },
  },
};
