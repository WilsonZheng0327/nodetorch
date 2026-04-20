import type { NodeDefinition } from '../../../core/nodedef';

export const agNewsNode: NodeDefinition = {
  type: 'data.ag_news',
  version: 1,
  displayName: 'AG News',
  description: 'News classification, 4 classes',
  category: ['Data', 'Text'],
  learnMore: 'News articles classified into 4 categories: World, Sports, Business, and Science/Technology. Tests whether the model can understand topic-level meaning from text, not just sentiment.',

  getProperties: () => [
    { id: 'batchSize', name: 'Batch Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 32, affects: 'execution', help: 'Number of samples per training step. Larger = faster but more memory. Common: 32, 64, 128.' },
    { id: 'maxLen', name: 'Max Length', type: { kind: 'number', min: 16, max: 512, integer: true }, defaultValue: 128, affects: 'execution', help: 'Maximum sequence length in tokens. Longer sequences use more memory. Texts beyond this are truncated.' },
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
            datasetType: 'data.ag_news',
            shapes: [
              { label: 'Tokens', value: [B, L] },
              { label: 'Labels', value: [B] },
              { label: 'Classes', value: '4 (World/Sports/Business/Sci-Tech)' },
            ],
          },
        };
      },
    },
  },
};
