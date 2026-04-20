// ReLU activation node.
// Shape passes through unchanged — ReLU doesn't alter dimensions.

import type { NodeDefinition } from '../../../core/nodedef';

export const reluNode: NodeDefinition = {
  type: 'ml.activations.relu',
  version: 1,
  displayName: 'ReLU',
  description: 'Rectified Linear Unit activation',
  category: ['ML', 'Activations'],
  learnMore: 'The most common activation function. Outputs the input if positive, zero if negative. Simple and fast, but can cause "dying neurons" \u2014 if a neuron always gets negative input, it permanently outputs zero and stops learning.',

  getProperties: () => [],

  getPorts: () => [
    {
      id: 'in',
      name: 'Input',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'out',
      name: 'Output',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const input = inputs.in;
        if (!input) {
          return { outputs: {} };
        }
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
