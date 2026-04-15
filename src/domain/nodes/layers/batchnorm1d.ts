import type { NodeDefinition } from '../../../core/nodedef';

export const batchNorm1dNode: NodeDefinition = {
  type: 'ml.layers.batchnorm1d',
  version: 1,
  displayName: 'BatchNorm1d',
  description: 'Batch normalization for 2D/3D input (linear layers)',
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

        const numFeatures = input[1];
        return {
          outputs: { out: input },
          metadata: {
            outputShape: input,
            paramCount: numFeatures * 2,
            paramBreakdown: `gamma: ${numFeatures} + beta: ${numFeatures} = ${numFeatures * 2}`,
            shapes: [{ label: 'Output', value: input }],
          },
        };
      },
    },
  },
};
