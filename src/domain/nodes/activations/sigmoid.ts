import type { NodeDefinition } from '../../../core/nodedef';

export const sigmoidNode: NodeDefinition = {
  type: 'ml.activations.sigmoid',
  version: 1,
  displayName: 'Sigmoid',
  description: 'Sigmoid activation (outputs 0-1)',
  category: ['ML', 'Activations'],
  learnMore: 'Squashes any value to the range (0, 1). Useful as the final activation for binary classification or when you need probabilities. Can cause vanishing gradients in deep networks because the gradient is very small for large/small inputs.',

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
        return { outputs: { out: input }, metadata: { outputShape: input, shapes: [{ label: 'Output', value: input }] } };
      },
    },
  },
};
