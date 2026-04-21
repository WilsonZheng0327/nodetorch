// Scaled Dot-Product Attention — the core attention operation.
// Q, K, V inputs. Computes: softmax(QK^T / sqrt(d_k)) V
// Simpler than MultiHeadAttention — no projection matrices, no heads.
// Input: Q [B, seq_q, d], K [B, seq_k, d], V [B, seq_k, d_v]
// Output: [B, seq_q, d_v]

import type { NodeDefinition } from '../../../core/nodedef';

export const attentionNode: NodeDefinition = {
  type: 'ml.layers.attention',
  version: 1,
  displayName: 'Attention',
  description: 'Scaled dot-product attention (QK^TV)',
  category: ['ML', 'Layers', 'Attention'],
  learnMore: 'Scaled dot-product attention \u2014 computes how much each position should attend to every other position. The attention weight between positions i and j is based on how similar their Query and Key vectors are. The foundation of all transformer architectures.',

  getProperties: () => [
    {
      id: 'dropout',
      name: 'Dropout',
      type: { kind: 'number', min: 0, max: 1, step: 0.1 },
      defaultValue: 0.0,
      affects: 'execution',
    },
    {
      id: 'causalMask',
      name: 'Causal Mask',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
      help: 'Prevents each position from attending to future positions. Required for autoregressive models (language models, GPT-style). Each token can only "see" tokens that came before it.',
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
      execute: async ({ inputs }) => {
        const query = inputs.query;
        const key = inputs.key;
        const value = inputs.value;

        if (!query) return { outputs: {} };

        if (query.length < 2) {
          return { outputs: {}, metadata: { error: `Query should be at least 2D, got [${query}]` } };
        }

        if (key && key[key.length - 1] !== query[query.length - 1]) {
          return { outputs: {}, metadata: { error: `Key embed dim ${key[key.length - 1]} != Query embed dim ${query[query.length - 1]}` } };
        }

        // Value seq_len must match Key seq_len (V is attended over same positions as K)
        if (key && value && key.length >= 2 && value.length >= 2) {
          const keySeq = key[key.length - 2];
          const valSeq = value[value.length - 2];
          if (keySeq !== valSeq) {
            return { outputs: {}, metadata: { error: `Value seq_len ${valSeq} != Key seq_len ${keySeq}` } };
          }
        }

        // Output: [...batch, seq_q, d_v]
        // d_v comes from value's last dim, seq_q from query's second-to-last
        const d_v = value ? value[value.length - 1] : query[query.length - 1];
        const outShape = [...query.slice(0, -1), d_v];

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'd_k', value: String(query[query.length - 1]) },
            ],
          },
        };
      },
    },
  },
};
