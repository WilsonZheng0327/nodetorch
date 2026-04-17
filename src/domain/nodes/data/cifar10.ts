import type { NodeDefinition } from '../../../core/nodedef';

export const cifar10Node: NodeDefinition = {
  type: 'data.cifar10',
  version: 1,
  displayName: 'CIFAR-10',
  description: '10-class color images, 32x32 RGB',
  category: ['Data'],

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
          outputs: { out: [B, 3, 32, 32], labels: [B] },
          metadata: {
            outputShape: [B, 3, 32, 32],
            datasetType: 'data.cifar10',
            shapes: [
              { label: 'Images', value: [B, 3, 32, 32] },
              { label: 'Labels', value: [B] },
            ],
          },
        };
      },
    },
  },
};
