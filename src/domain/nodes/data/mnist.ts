// MNIST data source node.
// Shape executor returns the dataset shape. No inputs — this is an entry point.

import type { NodeDefinition } from '../../../core/nodedef';

export const mnistNode: NodeDefinition = {
  type: 'data.mnist',
  version: 1,
  displayName: 'MNIST',
  description: 'Handwritten digits, 28x28 grayscale',
  category: ['Data', 'Image'],
  learnMore: 'The "hello world" of machine learning \u2014 70,000 handwritten digit images (0-9). Grayscale, 28\u00d728 pixels. Small enough to train quickly, complex enough to be interesting. If your model can\'t learn MNIST, something is fundamentally wrong.',

  getProperties: () => [
    {
      id: 'batchSize',
      name: 'Batch Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 1,
      affects: 'execution',
      help: 'Number of samples per training step. Larger = faster but more memory. Common: 32, 64, 128.',
    },
  ],

  getPorts: () => [
    {
      id: 'out',
      name: 'Images',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
    {
      id: 'labels',
      name: 'Labels',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ properties }) => {
        const B = properties.batchSize;
        return {
          outputs: {
            out: [B, 1, 28, 28],
            labels: [B],
          },
          metadata: {
            shapes: [
              { label: 'Images', value: [B, 1, 28, 28] },
              { label: 'Labels', value: [B] },
              { label: 'Classes', value: '10' },
              { label: 'Train', value: '60,000' },
              { label: 'Test', value: '10,000' },
              { label: 'Size', value: '~12 MB' },
            ],
            datasetType: 'data.mnist',
          },
        };
      },
    },
  },
};
