// Noise Input — random noise vector for generator input in GANs.
// No inputs — this is an entry point for the generator subgraph.
// The training loop injects fresh random noise each step.

import type { NodeDefinition } from '../../../core/nodedef';

export const noiseInputNode: NodeDefinition = {
  type: 'ml.gan.noise_input',
  version: 1,
  displayName: 'Noise Input',
  description: 'Random noise vector for generator input',
  category: ['ML', 'GAN'],
  color: '#f59e0b',

  getProperties: () => [
    {
      id: 'latentDim',
      name: 'Latent Dim',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 100,
      affects: 'execution' as const,
      help: 'Dimension of the random noise vector. Common: 100. Each training step generates fresh random noise.',
    },
    {
      id: 'batchSize',
      name: 'Batch Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 64,
      affects: 'execution' as const,
      help: 'Number of noise vectors per batch. Should match the dataset batch size.',
    },
  ],

  getPorts: () => [
    {
      id: 'out',
      name: 'Noise',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ properties }) => {
        const B = properties.batchSize ?? 64;
        const latentDim = properties.latentDim ?? 100;
        return {
          outputs: {
            out: [B, latentDim],
          },
          metadata: {
            shapes: [
              { label: 'Output', value: [B, latentDim] },
              { label: 'Latent Dim', value: String(latentDim) },
            ],
            outputShape: [B, latentDim],
          },
        };
      },
    },
  },
};
