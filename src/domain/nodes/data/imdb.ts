import type { NodeDefinition } from '../../../core/nodedef';

export const imdbNode: NodeDefinition = {
  type: 'data.imdb',
  version: 1,
  displayName: 'IMDb',
  description: 'Movie review sentiment, binary (positive/negative)',
  category: ['Data', 'Text'],
  learnMore: 'Movie reviews labeled as positive or negative sentiment. A standard benchmark for text classification. Reviews are tokenized into word indices \u2014 the model must learn which words and patterns indicate positive vs negative sentiment.',

  getProperties: () => [
    { id: 'batchSize', name: 'Batch Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 32, affects: 'execution', help: 'Number of samples per training step. Larger = faster but more memory. Common: 32, 64, 128.' },
    { id: 'maxLen', name: 'Max Length', type: { kind: 'number', min: 16, max: 512, integer: true }, defaultValue: 256, affects: 'execution', help: 'Maximum sequence length in tokens. Longer sequences use more memory. Texts beyond this are truncated.' },
    { id: 'vocabSize', name: 'Vocab Size', type: { kind: 'number', min: 100, integer: true }, defaultValue: 10000, affects: 'execution', help: 'Number of unique words to keep. Rare words beyond this are replaced with <unk>. 10000 covers most common words.' },
  ],

  getPorts: () => [
    { id: 'out', name: 'Token IDs', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
    { id: 'labels', name: 'Labels', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ properties }) => {
        const B = properties.batchSize;
        const L = properties.maxLen;
        return {
          outputs: { out: [B, L], labels: [B] },
          metadata: {
            outputShape: [B, L],
            datasetType: 'data.imdb',
            shapes: [
              { label: 'Tokens', value: [B, L] },
              { label: 'Labels', value: [B] },
              { label: 'Classes', value: '2 (pos/neg)' },
            ],
          },
        };
      },
    },
  },
};
