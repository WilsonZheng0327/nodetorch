// GAN Loss — adversarial loss for GAN training.
// Takes real_scores and fake_scores from the discriminator.
// Internally computes BCE with logits for both real and fake predictions.
// The training loop handles running the discriminator on both real and fake data.

import type { NodeDefinition } from '../../../core/nodedef';

export const ganLossNode: NodeDefinition = {
  type: 'ml.loss.gan',
  version: 1,
  displayName: 'GAN Loss',
  description: 'Adversarial loss for GAN training (handles both G and D losses internally)',
  category: ['ML', 'Loss'],
  color: '#ef4444',

  getProperties: () => [
    {
      id: 'labelSmoothing',
      name: 'Label Smoothing',
      type: { kind: 'number', min: 0, max: 0.5, step: 0.05 },
      defaultValue: 0.1,
      affects: 'execution' as const,
      help: 'Smooths the "real" label from 1.0 to (1-smoothing). Helps stabilize GAN training. 0 = no smoothing.',
    },
  ],

  getPorts: () => [
    {
      id: 'real_scores',
      name: 'Real Scores',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'fake_scores',
      name: 'Fake Scores',
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
      execute: async ({ inputs, properties }) => {
        const realScores = inputs.real_scores;
        const fakeScores = inputs.fake_scores;

        if (!realScores || !fakeScores) {
          return { outputs: {} };
        }

        const smoothing = properties.labelSmoothing ?? 0.1;

        return {
          outputs: { out: [] },
          metadata: {
            outputShape: ['scalar'],
            shapes: [
              { label: 'Output', value: 'scalar' },
              { label: 'Label Smoothing', value: String(smoothing) },
            ],
          },
        };
      },
    },
  },
};
