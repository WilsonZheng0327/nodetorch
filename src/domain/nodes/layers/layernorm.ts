// LayerNorm — normalizes across the last N dimensions.
// For transformers: input [B, seq_len, embed_dim] → normalized_shape = [embed_dim]
// For CNNs: input [B, C, H, W] → normalized_shape = [C, H, W]

import type { NodeDefinition } from '../../../core/nodedef';

export const layerNormNode: NodeDefinition = {
  type: 'ml.layers.layernorm',
  version: 1,
  displayName: 'LayerNorm',
  description: 'Layer normalization (used in transformers)',
  category: ['ML', 'Layers', 'Normalization'],

  getProperties: () => [
    {
      id: 'numLastDims',
      name: 'Normalized Dims',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 1,
      affects: 'execution',
    },
  ],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };

        const n = properties.numLastDims ?? 1;
        const normalizedShape = input.slice(-n);
        const paramCount = normalizedShape.reduce((a: number, b: number) => a * b, 1) * 2; // gamma + beta

        return {
          outputs: { out: input },
          metadata: {
            outputShape: input,
            paramCount,
            paramBreakdown: `gamma: [${normalizedShape}] + beta: [${normalizedShape}] = ${paramCount}`,
            shapes: [
              { label: 'Output', value: input },
              { label: 'Norm shape', value: normalizedShape },
            ],
          },
        };
      },
    },
  },
};
