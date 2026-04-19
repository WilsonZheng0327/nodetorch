import type { NodeDefinition } from '../../../core/nodedef';

export const dropout2dNode: NodeDefinition = {
  type: 'ml.layers.dropout2d',
  version: 1,
  displayName: 'Dropout2d',
  description: 'Spatial dropout — zeroes entire channels during training',
  category: ['ML', 'Layers', 'Regularization'],

  getProperties: () => [
    { id: 'p', name: 'Drop Probability', type: { kind: 'number', min: 0, max: 1, step: 0.1 }, defaultValue: 0.5, affects: 'execution', help: 'Probability of zeroing entire channels (not individual pixels). Better for CNNs because it drops spatial features as a unit.' },
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
