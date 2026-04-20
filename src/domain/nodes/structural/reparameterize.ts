// Reparameterize — VAE reparameterization trick.
// Samples z = mean + std * noise, where std = exp(0.5 * logvar).
// This allows gradients to flow through the sampling step.

import type { NodeDefinition } from '../../../core/nodedef';

export const reparameterizeNode: NodeDefinition = {
  type: 'ml.structural.reparameterize',
  version: 1,
  displayName: 'Reparameterize',
  description: 'VAE reparameterization trick: samples z = mean + std * noise',
  category: ['ML', 'Structural'],
  help: 'Samples from the learned distribution using the reparameterization trick. During training, adds random noise scaled by the variance. This allows gradients to flow through the sampling step.',

  getProperties: () => [],

  getPorts: () => [
    { id: 'mean', name: 'Mean', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'logvar', name: 'Log Variance', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const mean = inputs.mean;
        if (!mean) return { outputs: {} };

        return {
          outputs: { out: mean },
          metadata: {
            outputShape: mean,
            shapes: [{ label: 'Output', value: mean }],
          },
        };
      },
    },
  },
};
