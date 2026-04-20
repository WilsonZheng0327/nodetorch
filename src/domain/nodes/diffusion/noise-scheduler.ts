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
      name: 'Noise',
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

        // Input: [B, C, H, W] -> Output: [B, C+1, H, W] (extra timestep channel)
        const [B, C, H, W] = images;
        const outShape = [B, C + 1, H, W];
        // Noise output has same shape as input (no timestep channel)
        const noiseShape = [B, C, H, W];

        return {
          outputs: {
            out: outShape,
            noise: noiseShape,
          },
          metadata: {
            shapes: [
              { label: 'Input', value: images },
              { label: 'Noisy Output', value: outShape },
              { label: 'Noise Target', value: noiseShape },
            ],
            outputShape: outShape,
          },
        };
      },
    },
  },
};
