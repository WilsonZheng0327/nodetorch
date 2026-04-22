// MultiHeadAttention — scaled dot-product attention with multiple heads.
// For self-attention: connect the same source to Q, K, and V.
// For cross-attention: Q from one source, K/V from another.
// Input: [B, seq_len, embed_dim] for Q/K/V
// Output: [B, seq_len, embed_dim]

import type { NodeDefinition } from '../../../core/nodedef';

export const multiHeadAttentionNode: NodeDefinition = {
  type: 'ml.layers.multihead_attention',
  version: 1,
  displayName: 'MultiHeadAttention',
  description: 'Multi-head attention (self or cross)',
  category: ['ML', 'Layers', 'Attention'],
  learnMore: 'The core of transformer models. Each "head" learns to attend to different aspects of the input \u2014 one might focus on syntax, another on meaning. For self-attention, connect the same input to all three ports (Query, Key, Value). This lets each position in the sequence look at every other position.',

  getProperties: () => [
    {
      id: 'embedDim',
      name: 'Embed Dim',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 256,
      affects: 'execution',
      help: 'Dimension of input embeddings. Must match the embedding layer\'s output. Must be divisible by numHeads.',
    },
    {
      id: 'numHeads',
      name: 'Num Heads',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 8,
      affects: 'execution',
      help: 'Number of attention heads. Each head attends to different aspects of the input. Common: 4, 8. embedDim must be divisible by this.',
    },
    {
      id: 'dropout',
      name: 'Dropout',
      type: { kind: 'number', min: 0, max: 1, step: 0.1 },
      defaultValue: 0.0,
      affects: 'execution',
      help: 'Dropout applied to attention weights during training. Helps prevent overfitting.',
    },
    {
      id: 'causalMask',
      name: 'Causal Mask',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
      help: 'Prevents each position from attending to future positions. Required for autoregressive (GPT-style) language models. Each token can only "see" tokens that came before it.',
    },
  ],

  getPorts: () => [
    { id: 'query', name: 'Query', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'key', name: 'Key', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'value', name: 'Value', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const query = inputs.query;
        const key = inputs.key;

        if (!query) return { outputs: {} };

        const { embedDim, numHeads } = properties;

        if (embedDim % numHeads !== 0) {
          return { outputs: {}, metadata: { error: `embed_dim (${embedDim}) must be divisible by num_heads (${numHeads})` } };
        }

        if (query.length !== 3) {
          return { outputs: {}, metadata: { error: `Query should be [B, seq_len, embed_dim], got [${query}]` } };
        }

        if (key && key[key.length - 1] !== query[query.length - 1]) {
          return { outputs: {}, metadata: { error: `Key embed dim ${key[key.length - 1]} != Query embed dim ${query[query.length - 1]}` } };
        }

        // Output has same shape as query: [B, seq_q, embed_dim]
        const outShape = [query[0], query[1], embedDim];
        const headDim = embedDim / numHeads;

        // Params: 3 projection matrices (Q, K, V) + output projection, each with bias
        const paramCount = (3 * embedDim * embedDim + 3 * embedDim) + (embedDim * embedDim + embedDim);

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            paramCount,
            paramBreakdown: `Wq+Wk+Wv: 3×${embedDim}×${embedDim} + Wo: ${embedDim}×${embedDim} + biases`,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Heads', value: `${numHeads} × ${headDim}` },
            ],
          },
        };
      },
    },
  },
};
