import type { NodeDefinition } from '../../../core/nodedef';

export const instanceNorm2dNode: NodeDefinition = {
  type: 'ml.layers.instancenorm2d',
  version: 1,
  displayName: 'InstanceNorm2d',
  description: 'Instance normalization — normalizes each sample independently',
  category: ['ML', 'Layers', 'Normalization'],

  getProperties: () => [],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };
        if (input.length !== 4) return { outputs: {}, metadata: { error: 'InstanceNorm2d expects [B, C, H, W]' } };

        return {
          outputs: { out: input },
          metadata: {
            paramCount: input[1] * 2,
            outputShape: input,
            shapes: [{ label: 'Output', value: input }],
          },
        };
      },
    },
  },
};
