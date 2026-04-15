import type { NodeDefinition } from '../../../core/nodedef';

export const geluNode: NodeDefinition = {
  type: 'ml.activations.gelu',
  version: 1,
  displayName: 'GELU',
  description: 'Gaussian Error Linear Unit (used in transformers)',
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
