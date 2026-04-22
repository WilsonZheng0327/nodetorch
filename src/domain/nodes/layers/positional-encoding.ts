// PositionalEncoding — adds position information to embeddings.
// Self-attention is permutation-invariant: without positional info, "the cat sat"
// looks identical to "sat cat the". This node injects position into each embedding.
//
// Input:  [B, seq_len, embed_dim]
// Output: [B, seq_len, embed_dim] (same shape — positions added element-wise)

import type { NodeDefinition } from '../../../core/nodedef';

export const positionalEncodingNode: NodeDefinition = {
  type: 'ml.layers.positional_encoding',
  version: 1,
  displayName: 'PositionalEncoding',
  description: 'Adds position info to token embeddings (required for transformers)',
  category: ['ML', 'Layers'],
  learnMore: 'Self-attention treats input as a set of vectors — it has no idea which token came first. PositionalEncoding adds a unique signature for each position so the model knows the order. "Learned" trains a separate vector per position (like an extra Embedding). "Sinusoidal" uses fixed sine/cosine waves at different frequencies — no parameters, generalizes to longer sequences than seen in training. Either way, it gets added to the token embeddings before attention.',

  getProperties: () => [
    {
      id: 'maxLen',
      name: 'Max Length',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 512,
      affects: 'execution',
      help: 'Maximum sequence length the model can handle. Must be >= the dataset sequence length. Larger = more parameters (for learned encoding only).',
    },
    {
      id: 'encodingType',
      name: 'Type',
      type: {
        kind: 'select',
        options: [
          { value: 'learned', label: 'Learned' },
          { value: 'sinusoidal', label: 'Sinusoidal (no params)' },
        ],
      },
      defaultValue: 'learned',
      affects: 'execution',
      help: '"Learned": one trainable vector per position (used by GPT, BERT). "Sinusoidal": fixed sin/cos waves (the original Transformer paper). Both work; learned is slightly more flexible, sinusoidal generalizes to unseen lengths.',
    },
  ],

  getPorts: () => [
    { id: 'in', name: 'Embeddings', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Embeddings + Position', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };

        if (input.length !== 3) {
          return {
            outputs: {},
            metadata: { error: `Input should be [B, seq_len, embed_dim], got [${input.join(', ')}]` },
          };
        }

        const { maxLen, encodingType } = properties;
        const seqLen = input[1];
        const embedDim = input[2];

        if (typeof seqLen === 'number' && seqLen > maxLen) {
          return {
            outputs: {},
            metadata: { error: `Input seq_len (${seqLen}) > maxLen (${maxLen}). Increase maxLen.` },
          };
        }

        // Output shape is identical to input
        const outShape = input;
        const paramCount = encodingType === 'learned' ? maxLen * embedDim : 0;

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            paramCount,
            paramBreakdown: encodingType === 'learned'
              ? `position table: ${maxLen}×${embedDim} = ${paramCount.toLocaleString()}`
              : 'no parameters (fixed sinusoidal table)',
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Type', value: encodingType === 'learned' ? 'Learned' : 'Sinusoidal' },
              { label: 'Max length', value: String(maxLen) },
            ],
          },
        };
      },
    },
  },
};
