import type { NodeDefinition } from '../../../core/nodedef';

export const tanhNode: NodeDefinition = {
  type: 'ml.activations.tanh',
  version: 1,
  displayName: 'Tanh',
  description: 'Hyperbolic tangent activation (outputs -1 to 1)',
  category: ['ML', 'Activations'],

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
