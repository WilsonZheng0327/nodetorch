import type { NodeDefinition } from '../../../core/nodedef';

export const adamwNode: NodeDefinition = {
  type: 'ml.optimizers.adamw',
  version: 1,
  displayName: 'AdamW',
  description: 'AdamW optimizer (decoupled weight decay)',
  category: ['ML', 'Optimizers'],
  color: '#8b5cf6',

  getProperties: () => [
    {
      id: 'lr',
      name: 'Learning Rate',
      type: { kind: 'number', min: 0, step: 0.0001 },
      defaultValue: 0.001,
      affects: 'execution',
    },
    {
      id: 'beta1',
      name: 'Beta 1',
      type: { kind: 'number', min: 0, max: 1, step: 0.01 },
      defaultValue: 0.9,
      affects: 'execution',
    },
    {
      id: 'beta2',
      name: 'Beta 2',
      type: { kind: 'number', min: 0, max: 1, step: 0.001 },
      defaultValue: 0.999,
      affects: 'execution',
    },
    {
      id: 'weightDecay',
      name: 'Weight Decay',
      type: { kind: 'number', min: 0, step: 0.001 },
      defaultValue: 0.01,
      affects: 'execution',
    },
    {
      id: 'epochs',
      name: 'Epochs',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 2,
      affects: 'execution',
    },
  ],

  getPorts: () => [
    { id: 'loss', name: 'Loss', direction: 'input', dataType: 'scalar', allowMultiple: false, optional: false },
  ],

  executors: {},
};
