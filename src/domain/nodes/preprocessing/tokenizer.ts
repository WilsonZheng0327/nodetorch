// Tokenizer — configurable text tokenization as a graph node.
// Sits between a text data node and the embedding layer.
// Truncates/pads sequences to maxLen and caps vocabulary indices.
// Input: [B, seq_len] (token IDs from data node)
// Output: [B, maxLen] (normalized token IDs)

import type { NodeDefinition } from '../../../core/nodedef';

export const tokenizerNode: NodeDefinition = {
  type: 'ml.preprocessing.tokenizer',
  version: 1,
  displayName: 'Tokenizer',
  description: 'Text tokenization (character, word, or BPE)',
  category: ['ML', 'Preprocessing'],
  learnMore: 'Converts raw text into sequences of integer token IDs that a model can process. Character-level tokenization splits text into individual characters — small vocabulary (~100 tokens), simple but requires long sequences. Word-level splits on whitespace and punctuation — larger vocabulary (~10K–50K), captures whole words but struggles with rare/unseen words. BPE (Byte-Pair Encoding) learns common subword units — this is what GPT, BERT, and most modern models use. It balances vocabulary size with the ability to handle any word by breaking it into known pieces.',

  getProperties: () => [
    {
      id: 'mode',
      name: 'Mode',
      type: { kind: 'select', options: [
        { label: 'Character', value: 'character' },
        { label: 'Word', value: 'word' },
        { label: 'BPE', value: 'bpe' },
      ] },
      defaultValue: 'character',
      affects: 'execution',
      help: 'Tokenization strategy. Character: each character is one token (small vocab, ~100). Word: split on whitespace/punctuation (vocab ~10K). BPE: learned subword units (used by GPT, BERT — best for most tasks).',
    },
    {
      id: 'vocabSize',
      name: 'Vocab Size',
      type: { kind: 'number', min: 50, integer: true },
      defaultValue: 10000,
      affects: 'execution',
      help: 'Maximum vocabulary size. Tokens beyond this are mapped to <unk>. For character mode this is typically auto-determined (~100). For word mode, 10K–50K is common. For BPE, 30K–50K (GPT-2 uses 50257).',
    },
    {
      id: 'maxLen',
      name: 'Max Length',
      type: { kind: 'number', min: 1, max: 2048, integer: true },
      defaultValue: 256,
      affects: 'execution',
      help: 'Maximum sequence length in tokens. Longer sequences are truncated; shorter ones are padded with zeros. Memory scales with this value — 128–256 for character models, 256–512 for word/BPE.',
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

        const { mode, vocabSize, maxLen } = properties;

        // Input is [B, L] from data node → output is [B, maxLen]
        const B = input[0];
        const outShape = [B, maxLen];

        const modeLabels: Record<string, string> = {
          character: 'Character-level',
          word: 'Word-level',
          bpe: 'Byte-Pair Encoding',
        };

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Mode', value: modeLabels[mode] || mode },
              { label: 'Vocab Size', value: vocabSize.toLocaleString() },
            ],
          },
        };
      },
    },
  },
};
