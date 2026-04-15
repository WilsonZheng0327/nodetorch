// Permute — reorder tensor dimensions.
// Dimension order as comma-separated string. e.g., "0, 2, 1" swaps last two dims.

import type { NodeDefinition } from '../../../core/nodedef';

export const permuteNode: NodeDefinition = {
  type: 'ml.structural.permute',
  version: 1,
  displayName: 'Permute',
  description: 'Reorder tensor dimensions',
  category: ['ML', 'Structural'],

  getProperties: () => [
    {
      id: 'dims',
      name: 'Dimension Order',
      type: { kind: 'string' },
      defaultValue: '0, 2, 1',
      affects: 'execution',
    },
  ],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };

        const dims = (properties.dims ?? '0, 2, 1').split(',').map((s: string) => parseInt(s.trim()));

        if (dims.length !== input.length) {
          return { outputs: {}, metadata: { error: `Permute dims [${dims}] doesn't match input rank ${input.length}` } };
        }

        const outShape = dims.map((d: number) => {
          if (d < 0 || d >= input.length) return -1;
          return input[d];
        });

        if (outShape.includes(-1)) {
          return { outputs: {}, metadata: { error: `Invalid dimension index in [${dims}]` } };
        }

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
