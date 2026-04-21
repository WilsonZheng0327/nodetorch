// CrossEntropyLoss node.
// Takes predictions [B, C] and labels [B], outputs scalar loss.
// Applies softmax internally — no need for a separate Softmax node before this.

import type { NodeDefinition } from '../../../core/nodedef';

export const crossEntropyLossNode: NodeDefinition = {
  type: 'ml.loss.cross_entropy',
  version: 1,
  displayName: 'CrossEntropyLoss',
  description: 'Cross-entropy loss (includes softmax)',
  category: ['ML', 'Loss'],
  color: '#ef4444',
  learnMore: 'Measures how wrong the model\'s class predictions are. Combines softmax and negative log-likelihood in one efficient computation. Output of 0 means perfect predictions. The standard loss for classification tasks (image classification, text classification, etc.).',

  getProperties: () => [],

  getPorts: () => [
    {
      id: 'predictions',
      name: 'Predictions',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'labels',
      name: 'Labels',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'out',
      name: 'Loss',
      direction: 'output',
      dataType: 'scalar',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const predictions = inputs.predictions;
        const labels = inputs.labels;

        if (!predictions || !labels) {
          return { outputs: {} };
        }

        // Validate: predictions should be [B, C] or [B, seq_len, C]
        if (predictions.length !== 2 && predictions.length !== 3) {
          return {
            outputs: {},
            metadata: { error: `Predictions should be [B, C] or [B, seq_len, C], got [${predictions.join(', ')}]` },
          };
        }

        const B = predictions[0];
        const C = predictions[predictions.length - 1];
        const isSequence = predictions.length === 3;

        if (isSequence) {
          // [B, seq_len, C] with labels [B, seq_len]
          if (labels.length !== 2 || labels[0] !== B || labels[1] !== predictions[1]) {
            return {
              outputs: {},
              metadata: { error: `Labels should be [${B}, ${predictions[1]}], got [${labels.join(', ')}]` },
            };
          }
        } else {
          // [B, C] with labels [B]
          if (labels.length !== 1 || labels[0] !== B) {
            return {
              outputs: {},
              metadata: { error: `Labels batch size ${labels[0]} doesn't match predictions batch size ${B}` },
            };
          }
        }

        const shapes = [
          { label: 'Output', value: 'scalar' },
          { label: 'Classes', value: String(C) },
          { label: 'Batch', value: String(B) },
        ];
        if (isSequence) {
          shapes.push({ label: 'Seq length', value: String(predictions[1]) });
        }

        return {
          outputs: { out: [] }, // scalar — empty shape
          metadata: { outputShape: ['scalar'], shapes },
        };
      },
    },
  },
};
