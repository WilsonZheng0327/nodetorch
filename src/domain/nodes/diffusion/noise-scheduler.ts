// Noise Scheduler — manages the diffusion noise schedule for training.
// Adds noise to clean images at a random timestep t.
// Output: noisy images with an extra timestep channel, plus the actual noise for loss computation.

import type { NodeDefinition } from '../../../core/nodedef';

export const noiseSchedulerNode: NodeDefinition = {
  type: 'ml.diffusion.noise_scheduler',
  version: 1,
  displayName: 'Noise Scheduler',
  description: 'Diffusion noise schedule — adds noise at random timesteps during training',
  category: ['ML', 'Diffusion'],
  color: '#8b5cf6',

  getProperties: () => [
    {
      id: 'numTimesteps',
      name: 'Timesteps',
      type: { kind: 'number', min: 10, max: 1000, integer: true },
      defaultValue: 100,
      affects: 'execution' as const,
      help: 'Total denoising steps. More = better quality but slower training/sampling. 100 is good for learning, 1000 for production.',
    },
    {
      id: 'betaStart',
      name: 'Beta Start',
      type: { kind: 'number', min: 0.0001, max: 0.1, step: 0.0001 },
      defaultValue: 0.0001,
      affects: 'execution' as const,
      help: 'Noise level at t=0 (least noisy). Standard: 0.0001.',
    },
    {
      id: 'betaEnd',
      name: 'Beta End',
      type: { kind: 'number', min: 0.001, max: 0.1, step: 0.001 },
      defaultValue: 0.02,
      affects: 'execution' as const,
      help: 'Noise level at t=T (most noisy). Standard: 0.02.',
    },
    {
      id: 'scheduleType',
      name: 'Schedule',
      type: {
        kind: 'select',
        options: [
          { label: 'Linear', value: 'linear' },
          { label: 'Cosine', value: 'cosine' },
        ],
      },
      defaultValue: 'linear',
      affects: 'execution' as const,
      help: 'Linear schedule adds noise uniformly. Cosine schedule preserves more detail at early timesteps.',
    },
  ],

  getPorts: () => [
    {
      id: 'images',
      name: 'Images',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'out',
      name: 'Noisy Images',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
    {
      id: 'noise',
      name: 'Noise Target',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
    {
      id: 'timestep',
      name: 'Timestep Channel',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const images = inputs.images;
        if (!images || !Array.isArray(images) || images.length < 4) {
          return { outputs: {} };
        }

        const [B, C, H, W] = images;
        // Noisy images: same shape as input
        const outShape = [B, C, H, W];
        // Noise target: same shape
        const noiseShape = [B, C, H, W];
        // Timestep channel: [B, 1, H, W] — to be concatenated with noisy images
        const timestepShape = [B, 1, H, W];

        return {
          outputs: {
            out: outShape,
            noise: noiseShape,
            timestep: timestepShape,
          },
          metadata: {
            shapes: [
              { label: 'Noisy', value: outShape },
              { label: 'Noise', value: noiseShape },
              { label: 'Timestep', value: timestepShape },
            ],
            outputShape: outShape,
          },
        };
      },
    },
  },
};
