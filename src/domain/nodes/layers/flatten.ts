// Flatten node.
// Collapses all dimensions after batch into a single dimension.
// e.g., [B, 64, 7, 7] → [B, 3136]

import type { NodeDefinition } from '../../../core/nodedef';

export const flattenNode: NodeDefinition = {
  type: 'ml.layers.flatten',
  version: 1,
  displayName: 'Flatten',
  description: 'Flatten all dimensions after batch into one',
  category: ['ML', 'Layers'],
  learnMore: 'Collapses spatial dimensions into a single vector. Needed between convolutional layers (which output 3D feature maps) and linear layers (which expect 1D vectors). For example, [batch, 64, 7, 7] becomes [batch, 3136].',

  getProperties: () => [],

  getPorts: () => [
    {
      id: 'in',
      name: 'Input',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'out',
      name: 'Output',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const input = inputs.in;
        if (!input) {
          return { outputs: {} };
        }

        // Multiply all dimensions after batch
        const batch = input[0];
        const flat = input.slice(1).reduce((a: number, b: number) => a * b, 1);

        return {
          outputs: { out: [batch, flat] },
          metadata: {
            outputShape: [batch, flat],
            shapes: [{ label: 'Output', value: [batch, flat] }],
          },
        };
      },
    },
  },
};
