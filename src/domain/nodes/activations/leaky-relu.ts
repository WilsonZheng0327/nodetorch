import type { NodeDefinition } from '../../../core/nodedef';

export const leakyReluNode: NodeDefinition = {
  type: 'ml.activations.leaky_relu',
  version: 1,
  displayName: 'LeakyReLU',
  description: 'Leaky ReLU (small slope for negatives)',
  category: ['ML', 'Activations'],
  learnMore: 'Like ReLU but allows a small gradient for negative values instead of zero. Prevents the "dying ReLU" problem where neurons permanently stop learning. The negative slope is typically 0.01 or 0.2.',

  getProperties: () => [
    {
      id: 'negativeSlope',
      name: 'Negative Slope',
      type: { kind: 'number', min: 0, max: 1, step: 0.01 },
      defaultValue: 0.01,
      affects: 'execution',
      help: 'Slope for negative values. 0.01 = standard LeakyReLU. Higher values let more gradient flow through for negative inputs.',
    },
  ],

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
