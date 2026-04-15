import type { NodeDefinition } from '../../../core/nodedef';

export const dropoutNode: NodeDefinition = {
  type: 'ml.layers.dropout',
  version: 1,
  displayName: 'Dropout',
  description: 'Randomly zeros elements during training',
  category: ['ML', 'Layers', 'Regularization'],

  getProperties: () => [
    {
      id: 'p',
      name: 'Drop Probability',
      type: { kind: 'number', min: 0, max: 1, step: 0.1 },
      defaultValue: 0.5,
      affects: 'execution',
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

        return {
          outputs: { out: input },
          metadata: {
            outputShape: input,
            shapes: [{ label: 'Output', value: input }],
          },
        };
      },
    },
  },
};
