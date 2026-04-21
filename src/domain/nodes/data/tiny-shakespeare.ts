import type { NodeDefinition } from '../../../core/nodedef';

export const tinyShakespeareNode: NodeDefinition = {
  type: 'data.tiny_shakespeare',
  version: 1,
  displayName: 'Tiny Shakespeare',
  description: 'Character-level language modeling on Shakespeare (~1MB)',
  category: ['Data', 'Text'],
  learnMore: 'A small corpus of Shakespeare plays used for character-level language modeling. The model learns to predict the next character given previous characters. This is the simplest form of autoregressive text generation — the same principle behind GPT, but at the character level instead of word/subword tokens.',

  getProperties: () => [
    { id: 'batchSize', name: 'Batch Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 64, affects: 'execution', help: 'Number of sequences per training step.' },
    { id: 'seqLen', name: 'Sequence Length', type: { kind: 'number', min: 16, max: 512, integer: true }, defaultValue: 128, affects: 'execution', help: 'Number of characters per sequence. Longer = more context for the model but more memory. 128 is a good balance for character-level models.' },
  ],

  getPorts: () => [
    { id: 'out', name: 'Input Chars', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
    { id: 'labels', name: 'Target Chars', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ properties }) => {
        const B = properties.batchSize;
        const L = properties.seqLen;
        return {
          outputs: { out: [B, L], labels: [B, L] },
          metadata: {
            outputShape: [B, L],
            datasetType: 'data.tiny_shakespeare',
            shapes: [
              { label: 'Input', value: [B, L] },
              { label: 'Target', value: [B, L] },
              { label: 'Vocab', value: '65 chars' },
              { label: 'Task', value: 'Next-char prediction' },
            ],
          },
        };
      },
    },
  },
};
