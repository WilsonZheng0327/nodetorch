// Character Tokenizer — one token per character.
// Vocabulary is determined entirely by the upstream corpus (every unique
// character that appears in the text). Small vocab (typically <200) but
// produces long sequences.
// Input: [B, seq_len] (token IDs from data node)
// Output: [B, maxLen] (truncated/padded token IDs)

import type { NodeDefinition } from '../../../core/nodedef';

export const tokenizerCharNode: NodeDefinition = {
  type: 'ml.preprocessing.tokenizer_char',
  version: 1,
  displayName: 'Char Tokenizer',
  description: 'Character-level tokenization (vocab = unique chars in corpus)',
  category: ['ML', 'Preprocessing'],
  learnMore: 'Each character is a single token. The vocabulary is every unique character that appears in the upstream corpus — typically under 200 for English text, more for Unicode-heavy corpora. There is no vocab-size knob to set; click this node to open the detail panel and see the actual vocabulary after a dataset is connected. Small vocab is easy to learn, but sequences become very long. Classic char-level RNN/LM setups (Karpathy\'s char-rnn) use this.',

  getProperties: () => [
    {
      id: 'maxLen',
      name: 'Max Length',
      type: { kind: 'number', min: 1, max: 2048, integer: true },
      defaultValue: 256,
      affects: 'execution',
      help: 'Maximum sequence length in tokens. Longer sequences are truncated; shorter ones are right-padded with 0. Memory scales with this value.',
    },
    {
      id: 'lowercase',
      name: 'Lowercase',
      type: { kind: 'boolean' },
      defaultValue: false,
      affects: 'execution',
      help: 'Lowercase the input before tokenizing. When off, capital and lowercase letters get separate tokens (e.g. "T" and "t" are different).',
    },
  ],

  getPorts: () => [
    { id: 'in', name: 'Text Data', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Token IDs', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };
        const { maxLen } = properties;
        const B = input[0];
        const outShape = [B, maxLen];
        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Mode', value: 'Character-level' },
              { label: 'Vocab', value: 'from corpus' },
            ],
          },
        };
      },
    },
  },
};
