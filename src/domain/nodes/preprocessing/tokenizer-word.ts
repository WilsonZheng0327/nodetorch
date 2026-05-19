// Word Tokenizer — split on whitespace (and optionally punctuation), keep
// the top-K most frequent tokens, map everything else to <unk>.
// Input: [B, seq_len] (token IDs from data node)
// Output: [B, maxLen] (truncated/padded token IDs)

import type { NodeDefinition } from '../../../core/nodedef';

export const tokenizerWordNode: NodeDefinition = {
  type: 'ml.preprocessing.tokenizer_word',
  version: 1,
  displayName: 'Word Tokenizer',
  description: 'Word-level tokenization (frequency-based vocab)',
  category: ['ML', 'Preprocessing'],
  learnMore: 'Splits text on whitespace into word tokens. The vocab is the top-K most frequent words in the corpus; everything else (rare words, words unseen at training time) becomes <unk>. Larger vocab captures more words but inflates the embedding table. Cannot represent out-of-vocabulary words at inference time — that\'s what subword tokenizers (BPE) fix.',

  getProperties: () => [
    {
      id: 'vocabSize',
      name: 'Vocab Size',
      type: { kind: 'number', min: 50, integer: true },
      defaultValue: 10000,
      affects: 'execution',
      help: 'Maximum number of word types to keep (most frequent first). Everything else maps to <unk>. 10K–50K is typical for English.',
    },
    {
      id: 'maxLen',
      name: 'Max Length',
      type: { kind: 'number', min: 1, max: 2048, integer: true },
      defaultValue: 256,
      affects: 'execution',
      help: 'Maximum sequence length in tokens. Longer sequences are truncated; shorter ones are right-padded with 0.',
    },
    {
      id: 'lowercase',
      name: 'Lowercase',
      type: { kind: 'boolean' },
      defaultValue: true,
      affects: 'execution',
      help: 'Lowercase the input before tokenizing. When on, "The" and "the" share one token (smaller vocab, less data needed). When off, they\'re separate.',
    },
    {
      id: 'splitPunctuation',
      name: 'Split Punctuation',
      type: { kind: 'boolean' },
      defaultValue: true,
      affects: 'execution',
      help: 'Treat punctuation as its own tokens. When on, "don\'t" → ["don", "\'", "t"]. When off, "don\'t" stays one token (but then "don\'t" and "don\'t." are different tokens, etc.).',
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
        const { vocabSize, maxLen } = properties;
        const B = input[0];
        const outShape = [B, maxLen];
        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Mode', value: 'Word-level' },
              { label: 'Vocab Size', value: vocabSize.toLocaleString() },
            ],
          },
        };
      },
    },
  },
};
