// Linear (fully connected) layer node.
// Takes the last dimension of the input and maps it to outFeatures.

import type { NodeDefinition } from '../../../core/nodedef';

export const linearNode: NodeDefinition = {
  type: 'ml.layers.linear',
  version: 1,
  displayName: 'Linear',
  description: 'Fully connected layer',
  category: ['ML', 'Layers'],

  getProperties: () => [
    {
      id: 'outFeatures',
      name: 'Out Features',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 128,
      affects: 'execution',
    },
  ],

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
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) {
          return { outputs: {} };
        }

        // Input can be any shape — Linear operates on the last dimension
        const inFeatures = input[input.length - 1];
        const outFeatures = properties.outFeatures;
        const outShape = [...input.slice(0, -1), outFeatures];

        return {
          outputs: { out: outShape },
          metadata: {
            paramCount: inFeatures * outFeatures + outFeatures,
            paramBreakdown: `weights: ${inFeatures}x${outFeatures} = ${inFeatures * outFeatures}  +  bias: ${outFeatures}  =  ${inFeatures * outFeatures + outFeatures}`,
            outputShape: outShape,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Weights', value: [inFeatures, outFeatures] },
              { label: 'Bias', value: [outFeatures] },
            ],
          },
        };
      },
    },
  },
};
