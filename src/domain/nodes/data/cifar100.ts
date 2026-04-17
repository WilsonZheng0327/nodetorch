// CIFAR-100 data source node.
// 100 fine-grained classes grouped into 20 superclasses.
// 32x32 color images.

import type { NodeDefinition } from '../../../core/nodedef';

export const cifar100Node: NodeDefinition = {
  type: 'data.cifar100',
  version: 1,
  displayName: 'CIFAR-100',
  description: '100-class color images, 32x32 RGB',
  category: ['Data'],

  getProperties: () => [
    {
      id: 'batchSize',
      name: 'Batch Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 32,
      affects: 'execution',
    },
    {
      id: 'useCoarseLabels',
      name: 'Use Coarse Labels',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
    },
    {
      id: 'augHFlip',
      name: 'Random H-Flip',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
    },
    {
      id: 'augRandomCrop',
      name: 'Random Crop (4px pad)',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
    },
    {
      id: 'augColorJitter',
      name: 'Color Jitter',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
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
        const coarse = properties.useCoarseLabels;

        return {
          outputs: {
            out: [B, 3, 32, 32],
            labels: [B],
          },
          metadata: {
            shapes: [
              { label: 'Images', value: [B, 3, 32, 32] },
              { label: 'Labels', value: [B] },
              { label: 'Classes', value: coarse ? '20' : '100' },
              { label: 'Train', value: '50,000' },
              { label: 'Test', value: '10,000' },
              { label: 'Size', value: '~161 MB' },
            ],
            datasetType: 'data.cifar100',
          },
        };
      },
    },
  },
};
