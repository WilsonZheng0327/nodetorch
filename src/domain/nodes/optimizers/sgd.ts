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
  learnMore: 'The classic optimizer \u2014 updates weights by subtracting the gradient scaled by the learning rate. With momentum, it also considers past gradients to smooth the updates and escape local minima. Simple, well-understood, and often the best choice for CNNs with careful learning rate tuning.',

  getProperties: () => [
    {
      id: 'lr',
      name: 'Learning Rate',
      type: { kind: 'number', min: 0, step: 0.001 },
      defaultValue: 0.01,
      affects: 'execution',
      help: 'Step size for each weight update. Too high → unstable training. Too low → slow convergence. Start with 0.01 for SGD, 0.001 for Adam.',
    },
    {
      id: 'momentum',
      name: 'Momentum',
      type: { kind: 'number', min: 0, max: 1, step: 0.1 },
      defaultValue: 0.9,
      affects: 'execution',
      help: 'Accumulates past gradients to smooth updates and escape local minima. 0 = no momentum (pure SGD). 0.9 is the standard choice.',
    },
    {
      id: 'weightDecay',
      name: 'Weight Decay',
      type: { kind: 'number', min: 0, step: 0.0001 },
      defaultValue: 0,
      affects: 'execution',
      help: 'L2 regularization — penalizes large weights to reduce overfitting. Try 1e-4 to 5e-4. 0 = no regularization.',
    },
    {
      id: 'epochs',
      name: 'Epochs',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 2,
      affects: 'execution',
      help: 'Number of full passes through the training data. More epochs = more training time but potentially better accuracy (until overfitting).',
    },
    {
      id: 'valSplit',
      name: 'Val Split',
      type: { kind: 'number', min: 0, max: 0.5, step: 0.05 },
      defaultValue: 0.1,
      affects: 'execution',
      help: 'Fraction of training data held out for validation. Used to detect overfitting. 0.1 = 10% validation, 90% training. Set to 0 to use all data for training.',
    },
    {
      id: 'seed',
      name: 'Random Seed',
      type: { kind: 'number', min: 0, integer: true },
      defaultValue: 42,
      affects: 'execution',
      help: 'Controls random initialization and data shuffling. Same seed = reproducible results.',
    },
    {
      id: 'scheduler',
      name: 'LR Scheduler',
      type: {
        kind: 'select',
        options: [
          { label: 'None (constant)', value: 'none' },
          { label: 'Step decay', value: 'step' },
          { label: 'Cosine annealing', value: 'cosine' },
          { label: 'Linear warmup', value: 'warmup' },
        ],
      },
      defaultValue: 'none',
      affects: 'execution',
      help: 'Adjusts the learning rate during training. Cosine annealing gradually reduces LR to near zero. Step decay halves LR every few epochs. Warmup starts low and ramps up.',
    },
    {
      id: 'earlyStopPatience',
      name: 'Early Stop Patience',
      type: { kind: 'number', min: 0, integer: true },
      defaultValue: 0,
      affects: 'execution',
      help: 'Stop training if validation loss doesn\'t improve for this many epochs. 0 = disabled. Try 5-10 to prevent overfitting.',
    },
    {
      id: 'gradClip',
      name: 'Grad Clip Norm',
      type: { kind: 'number', min: 0, step: 0.1 },
      defaultValue: 0,
      affects: 'execution',
      help: 'Clips gradient norm to this value to prevent exploding gradients. Useful for RNNs. 0 = no clipping. Try 1.0-5.0.',
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
