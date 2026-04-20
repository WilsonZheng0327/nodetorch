// Timestep Embedding — sinusoidal embedding of the diffusion timestep.
// No input port: the training loop injects the timestep directly.
// Present in the graph so the training loop can find and use it.

import type { NodeDefinition } from '../../../core/nodedef';

export const timestepEmbedNode: NodeDefinition = {
  type: 'ml.diffusion.timestep_embed',
  version: 1,
  displayName: 'Timestep Embed',
  description: 'Sinusoidal timestep embedding for diffusion models (injected by training loop)',
  category: ['ML', 'Diffusion'],
  color: '#8b5cf6',

  getProperties: () => [
    {
      id: 'embedDim',
      name: 'Embed Dim',
      type: { kind: 'number', min: 16, integer: true },
      defaultValue: 128,
      affects: 'execution' as const,
      help: 'Dimension of the timestep embedding vector. Must match a layer downstream (e.g., gets concatenated or added to features).',
    },
  ],

  getPorts: () => [
    {
      id: 'out',
      name: 'Embedding',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ properties }) => {
        const embedDim = properties.embedDim ?? 128;
        return {
          outputs: {
            out: [1, embedDim],
          },
          metadata: {
            shapes: [
              { label: 'Output', value: [1, embedDim] },
              { label: 'Embed Dim', value: String(embedDim) },
            ],
            outputShape: [1, embedDim],
          },
        };
      },
    },
  },
};
