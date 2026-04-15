// Embedding — lookup table that maps integer indices to dense vectors.
// Input: [B, seq_len] of integer token IDs
// Output: [B, seq_len, embedding_dim]

import type { NodeDefinition } from '../../../core/nodedef';

export const embeddingNode: NodeDefinition = {
  type: 'ml.layers.embedding',
  version: 1,
  displayName: 'Embedding',
  description: 'Token embedding lookup (maps indices to vectors)',
  category: ['ML', 'Layers'],

  getProperties: () => [
    {
      id: 'numEmbeddings',
      name: 'Vocab Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 10000,
      affects: 'execution',
    },
    {
      id: 'embeddingDim',
      name: 'Embedding Dim',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 256,
      affects: 'execution',
    },
  ],

  getPorts: () => [
    { id: 'in', name: 'Token IDs', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Embeddings', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };

        const { numEmbeddings, embeddingDim } = properties;
        // Input: [B, seq_len] → Output: [B, seq_len, embedding_dim]
        const outShape = [...input, embeddingDim];
        const paramCount = numEmbeddings * embeddingDim;

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            paramCount,
            paramBreakdown: `table: ${numEmbeddings}x${embeddingDim} = ${paramCount.toLocaleString()}`,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Table', value: [numEmbeddings, embeddingDim] },
            ],
          },
        };
      },
    },
  },
};
