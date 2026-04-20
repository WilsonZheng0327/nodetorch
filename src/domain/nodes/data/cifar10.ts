import type { NodeDefinition } from '../../../core/nodedef';

export const cifar10Node: NodeDefinition = {
  type: 'data.cifar10',
  version: 1,
  displayName: 'CIFAR-10',
  description: '10-class color images, 32x32 RGB',
  category: ['Data', 'Image'],
  learnMore: 'Ten classes of small color photographs (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck). 32\u00d732 RGB images \u2014 much harder than MNIST because of complex backgrounds, varying poses, and color. A standard benchmark for CNN architectures.',

  getProperties: () => [
    { id: 'batchSize', name: 'Batch Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 32, affects: 'execution', help: 'Number of samples per training step. Larger = faster but more memory. Common: 32, 64, 128.' },
    { id: 'augHFlip', name: 'Random H-Flip', type: { kind: 'boolean' }, defaultValue: false, affects: 'execution', help: 'Randomly flip images horizontally. Simple augmentation that doubles effective dataset size.' },
    { id: 'augRandomCrop', name: 'Random Crop (4px pad)', type: { kind: 'boolean' }, defaultValue: false, affects: 'execution', help: 'Randomly crop with 4px padding. Forces the model to handle slight position shifts.' },
    { id: 'augColorJitter', name: 'Color Jitter', type: { kind: 'boolean' }, defaultValue: false, affects: 'execution', help: 'Randomly adjust brightness, contrast, and saturation. Helps generalize to varying lighting.' },
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
