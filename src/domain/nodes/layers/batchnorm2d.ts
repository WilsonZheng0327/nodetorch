import type { NodeDefinition } from '../../../core/nodedef';

export const batchNorm2dNode: NodeDefinition = {
  type: 'ml.layers.batchnorm2d',
  version: 1,
  displayName: 'BatchNorm2d',
  description: 'Batch normalization over 4D input',
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
            // BN has 2 * num_features params (gamma + beta)
            paramCount: numFeatures * 2,
            paramBreakdown: `gamma: ${numFeatures} + beta: ${numFeatures} = ${numFeatures * 2}`,
            shapes: [{ label: 'Output', value: input }],
          },
        };
      },
    },
  },
};
