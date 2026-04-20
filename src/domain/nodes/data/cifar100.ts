// CIFAR-100 data source node.
// 100 fine-grained classes grouped into 20 superclasses.
// 32x32 color images.

import type { NodeDefinition } from '../../../core/nodedef';

export const cifar100Node: NodeDefinition = {
  type: 'data.cifar100',
  version: 1,
  displayName: 'CIFAR-100',
  description: '100-class color images, 32x32 RGB',
  category: ['Data', 'Image'],
  learnMore: 'Like CIFAR-10 but with 100 fine-grained classes (maple tree, oak tree, palm tree instead of just "tree"). Much harder because there are fewer examples per class and the visual differences between similar classes are subtle.',

  getProperties: () => [
    {
      id: 'batchSize',
      name: 'Batch Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 32,
      affects: 'execution',
      help: 'Number of samples per training step. Larger = faster but more memory. Common: 32, 64, 128.',
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
      help: 'Randomly flip images horizontally. Simple augmentation that doubles effective dataset size.',
    },
    {
      id: 'augRandomCrop',
      name: 'Random Crop (4px pad)',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
      help: 'Randomly crop with 4px padding. Forces the model to handle slight position shifts.',
    },
    {
      id: 'augColorJitter',
      name: 'Color Jitter',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
      help: 'Randomly adjust brightness, contrast, and saturation. Helps generalize to varying lighting.',
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
