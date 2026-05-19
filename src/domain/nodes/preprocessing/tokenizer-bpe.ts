// BPE Tokenizer — Byte-Pair Encoding, learned from the upstream corpus.
// Starts character-level, repeatedly merges the most frequent adjacent
// pair into a new token until vocab_size is reached.
// Input: [B, seq_len] (token IDs from data node)
// Output: [B, maxLen] (truncated/padded token IDs)

import type { NodeDefinition } from '../../../core/nodedef';

export const tokenizerBpeNode: NodeDefinition = {
  type: 'ml.preprocessing.tokenizer_bpe',
  version: 1,
  displayName: 'BPE Tokenizer',
  description: 'Byte-Pair Encoding (learned subword units)',
  category: ['ML', 'Preprocessing'],
  learnMore: 'BPE learns merge rules from your corpus: it starts with characters, then repeatedly merges the most-frequent adjacent pair into a new token. Common words end up as single tokens; rare words break into subword pieces. This is what GPT, BERT, and most modern LMs use. The end-of-word marker (default </w>) gets appended to every word so the tokenizer can tell "play" (start of word) from "play" (middle of word) — you\'ll see this marker in many of the learned tokens.',

  getProperties: () => [
    {
      id: 'vocabSize',
      name: 'Vocab Size',
      type: { kind: 'number', min: 50, integer: true },
      defaultValue: 10000,
      affects: 'execution',
      help: 'Target vocabulary size (including special tokens and base characters). BPE keeps merging pairs until reaching this size. GPT-2 uses 50257; for educational corpora, 1K–10K is plenty.',
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
      help: 'Lowercase the input before learning merges. When on, "The" and "the" produce the same tokens.',
    },
    {
      id: 'endOfWordMarker',
      name: 'End-of-Word Marker',
      type: { kind: 'string' },
      defaultValue: '</w>',
      affects: 'execution',
      help: 'Suffix appended to every word before BPE merging, so the tokenizer can distinguish a piece at the end of a word from the same piece mid-word. Default </w> follows the original BPE paper. Setting to empty disables the marker (treats words and word-fragments identically).',
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
              { label: 'Mode', value: 'Byte-Pair Encoding' },
              { label: 'Vocab Size', value: vocabSize.toLocaleString() },
            ],
          },
        };
      },
    },
  },
};
