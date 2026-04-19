import type { NodeDefinition } from '../../../core/nodedef';

export const softmaxNode: NodeDefinition = {
  type: 'ml.activations.softmax',
  version: 1,
  displayName: 'Softmax',
  description: 'Softmax activation (outputs sum to 1)',
  category: ['ML', 'Activations'],

  getProperties: () => [
    {
      id: 'dim',
      name: 'Dimension',
      type: { kind: 'number', min: -1, integer: true },
      defaultValue: -1,
      affects: 'execution',
      help: 'Dimension to apply softmax over. -1 (last dim) is standard for classification logits.',
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
