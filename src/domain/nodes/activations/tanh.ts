import type { NodeDefinition } from '../../../core/nodedef';

export const tanhNode: NodeDefinition = {
  type: 'ml.activations.tanh',
  version: 1,
  displayName: 'Tanh',
  description: 'Hyperbolic tangent activation (outputs -1 to 1)',
  category: ['ML', 'Activations'],
  learnMore: 'Squashes values to (-1, 1). Similar to Sigmoid but centered at zero, which often helps training. Commonly used as the final activation in GAN generators to produce images in the [-1, 1] range.',

  getProperties: () => [],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const input = inputs.in;
        if (!input || (typeof input === 'object' && !Array.isArray(input))) return { outputs: {} };
        return { outputs: { out: input }, metadata: { outputShape: input, shapes: [{ label: 'Output', value: input }] } };
      },
    },
  },
};
