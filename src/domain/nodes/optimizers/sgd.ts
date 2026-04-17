// SGD optimizer node.
// Not part of the data flow — the training loop reads its properties.
// Has a loss input to define what it's optimizing.

import type { NodeDefinition } from '../../../core/nodedef';

export const sgdNode: NodeDefinition = {
  type: 'ml.optimizers.sgd',
  version: 1,
  displayName: 'SGD',
  description: 'Stochastic Gradient Descent optimizer',
  category: ['ML', 'Optimizers'],
  color: '#8b5cf6',

  getProperties: () => [
    {
      id: 'lr',
      name: 'Learning Rate',
      type: { kind: 'number', min: 0, step: 0.001 },
      defaultValue: 0.01,
      affects: 'execution',
    },
    {
      id: 'momentum',
      name: 'Momentum',
      type: { kind: 'number', min: 0, max: 1, step: 0.1 },
      defaultValue: 0.9,
      affects: 'execution',
    },
    {
      id: 'weightDecay',
      name: 'Weight Decay',
      type: { kind: 'number', min: 0, step: 0.0001 },
      defaultValue: 0,
      affects: 'execution',
    },
    {
      id: 'epochs',
      name: 'Epochs',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 2,
      affects: 'execution',
    },
    {
      id: 'valSplit',
      name: 'Val Split',
      type: { kind: 'number', min: 0, max: 0.5, step: 0.05 },
      defaultValue: 0.1,
      affects: 'execution',
    },
  ],

  getPorts: () => [
    {
      id: 'loss',
      name: 'Loss',
      direction: 'input',
      dataType: 'scalar',
      allowMultiple: false,
      optional: false,
    },
  ],

  // No shape executor — optimizer doesn't participate in shape inference
  executors: {},
};
