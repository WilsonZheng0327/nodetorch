import type { NodeDefinition } from '../../../core/nodedef';

export const fashionMnistNode: NodeDefinition = {
  type: 'data.fashion_mnist',
  version: 1,
  displayName: 'FashionMNIST',
  description: 'Fashion items, 28x28 grayscale, 10 classes',
  category: ['Data', 'Image'],

  getProperties: () => [
    { id: 'batchSize', name: 'Batch Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 32, affects: 'execution' },
  ],

  getPorts: () => [
    { id: 'out', name: 'Images', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
    { id: 'labels', name: 'Labels', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ properties }) => {
        const B = properties.batchSize;
        return {
          outputs: { out: [B, 1, 28, 28], labels: [B] },
          metadata: {
            outputShape: [B, 1, 28, 28],
            datasetType: 'data.fashion_mnist',
            shapes: [
              { label: 'Images', value: [B, 1, 28, 28] },
              { label: 'Labels', value: [B] },
            ],
          },
        };
      },
    },
  },
};
