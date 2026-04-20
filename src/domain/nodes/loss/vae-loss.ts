// VAE Loss — combines reconstruction loss (MSE) and KL divergence.
// Takes 4 inputs: reconstruction, original, mean, logvar.
// beta controls the weight of KL divergence (beta=1 is standard VAE).

import type { NodeDefinition } from '../../../core/nodedef';

export const vaeLossNode: NodeDefinition = {
  type: 'ml.loss.vae',
  version: 1,
  displayName: 'VAE Loss',
  description: 'Reconstruction + KL divergence loss for VAEs',
  category: ['ML', 'Loss'],
  color: '#ef4444',
  learnMore: 'Combines reconstruction loss (how well the decoder recreates the input) with KL divergence (how close the latent distribution is to a standard normal). The \u03B2 parameter controls the trade-off \u2014 higher \u03B2 forces a more organized latent space but may sacrifice reconstruction quality.',

  getProperties: () => [
    {
      id: 'beta',
      name: 'Beta',
      type: { kind: 'number', min: 0, step: 0.1 },
      defaultValue: 1.0,
      affects: 'execution',
      help: 'Weight of the KL divergence term. \u03B2=1 is standard VAE. Higher \u03B2 encourages disentangled latent space. Lower \u03B2 prioritizes reconstruction quality.',
    },
  ],

  getPorts: () => [
    { id: 'reconstruction', name: 'Reconstruction', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'original', name: 'Original', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'mean', name: 'Mean', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'logvar', name: 'Log Variance', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Loss', direction: 'output', dataType: 'scalar', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const reconstruction = inputs.reconstruction;
        const original = inputs.original;

        if (!reconstruction || !original) return { outputs: {} };

        const beta = properties.beta ?? 1.0;

        return {
          outputs: { out: [] },
          metadata: {
            outputShape: ['scalar'],
            shapes: [
              { label: 'Output', value: 'scalar' },
              { label: 'Beta', value: String(beta) },
            ],
          },
        };
      },
    },
  },
};
