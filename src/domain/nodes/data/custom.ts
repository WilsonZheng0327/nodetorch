// Custom dataset source node — loads any HuggingFace dataset, described
// declaratively by its properties (mirrors the backend DatasetSpec). Two axes,
// input_kind × task, plus the HF source, column mapping, and per-kind transform
// params. The shape executor computes the output shape purely from properties —
// no backend round-trip. All the actual loading/interpreting happens in the
// backend (dataprep/custom_dataset.py), reached via the resolvers.

import type { NodeDefinition } from '../../../core/nodedef';

export const customNode: NodeDefinition = {
  type: 'data.custom',
  version: 1,
  displayName: 'Custom Dataset',
  description: 'Load any HuggingFace dataset',
  category: ['Data'],
  learnMore:
    'Bring your own data. Point this at a HuggingFace dataset (e.g. "mnist", "ag_news", "user/dataset"), map its columns to input and label, and pick how each sample is turned into a tensor (image / text / vector) and what the model predicts (classification / language model / regression / autoencoder). It flows through training, inference, and preview exactly like the built-in datasets.',

  getProperties: () => [
    // --- Source ---
    {
      id: 'hfId',
      name: 'HF Dataset',
      type: { kind: 'string' },
      defaultValue: '',
      group: 'Source',
      affects: 'execution',
      help: 'HuggingFace dataset id, e.g. "mnist", "ag_news", or "user/dataset".',
    },
    {
      id: 'hfConfig',
      name: 'Config',
      type: { kind: 'string' },
      defaultValue: '',
      group: 'Source',
      affects: 'execution',
      help: 'Optional dataset config / subset name (leave blank for the default).',
    },
    {
      id: 'trainSplit',
      name: 'Train Split',
      type: { kind: 'string' },
      defaultValue: 'train',
      group: 'Source',
      affects: 'execution',
      help: 'Name of the training split (usually "train").',
    },
    {
      id: 'testSplit',
      name: 'Test Split',
      type: { kind: 'string' },
      defaultValue: 'test',
      group: 'Source',
      affects: 'execution',
      help: 'Name of the held-out test split (blank if the dataset has none).',
    },
    {
      id: 'trustRemoteCode',
      name: 'Trust Remote Code',
      type: { kind: 'boolean' },
      defaultValue: false,
      group: 'Source',
      affects: 'execution',
      help: 'Allow datasets that run their own loader script. Off is safer; only enable for datasets you trust.',
    },

    // --- Task ---
    {
      id: 'inputKind',
      name: 'Input Kind',
      type: {
        kind: 'select',
        options: [
          { label: 'Image', value: 'image' },
          { label: 'Text', value: 'text' },
          { label: 'Vector', value: 'vector' },
        ],
      },
      defaultValue: 'image',
      group: 'Task',
      affects: 'both', // changes the output shape and which params are relevant
      help: 'How each sample is encoded into a tensor.',
    },
    {
      id: 'task',
      name: 'Task',
      type: {
        kind: 'select',
        options: [
          { label: 'Classification', value: 'classification' },
          { label: 'Language Model', value: 'language_model' },
          { label: 'Regression', value: 'regression' },
          { label: 'Autoencoder', value: 'autoencoder' },
        ],
      },
      defaultValue: 'classification',
      group: 'Task',
      affects: 'both',
      help: 'What the model predicts — the target formed from each sample.',
    },
    {
      id: 'batchSize',
      name: 'Batch Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 32,
      group: 'Task',
      affects: 'execution',
      help: 'Number of samples per training step.',
    },

    // --- Column mapping ---
    {
      id: 'inputColumn',
      name: 'Input Column',
      type: { kind: 'string' },
      defaultValue: 'image',
      group: 'Mapping',
      affects: 'execution',
      help: 'Dataset column holding the input (e.g. "image", "text", or a feature column).',
    },
    {
      id: 'labelColumn',
      name: 'Label Column',
      type: { kind: 'string' },
      defaultValue: 'label',
      group: 'Mapping',
      affects: 'execution',
      visible: (p) => p.task !== 'language_model',
      help: 'Dataset column holding the label / target.',
    },
    {
      id: 'classNames',
      name: 'Class Names',
      type: { kind: 'string' },
      defaultValue: '',
      group: 'Mapping',
      affects: 'execution',
      visible: (p) => p.task === 'classification',
      help: 'Optional comma-separated class labels, in class-id order. Blank = infer from the dataset.',
    },

    // --- Image params ---
    {
      id: 'imageSize',
      name: 'Image Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 32,
      group: 'Image',
      affects: 'both',
      visible: (p) => p.inputKind === 'image',
      help: 'Square resize target (height = width), in pixels.',
    },
    {
      id: 'channels',
      name: 'Channels',
      type: {
        kind: 'select',
        options: [
          { label: '1 (grayscale)', value: 1 },
          { label: '3 (RGB)', value: 3 },
        ],
      },
      defaultValue: 3,
      group: 'Image',
      affects: 'both',
      visible: (p) => p.inputKind === 'image',
      help: 'Number of image channels.',
    },
    {
      id: 'normalizeMean',
      name: 'Normalize Mean',
      type: { kind: 'string' },
      defaultValue: '',
      group: 'Image',
      affects: 'execution',
      visible: (p) => p.inputKind === 'image',
      help: 'Optional per-channel mean, comma-separated (e.g. "0.5,0.5,0.5"). Blank = no normalization.',
    },
    {
      id: 'normalizeStd',
      name: 'Normalize Std',
      type: { kind: 'string' },
      defaultValue: '',
      group: 'Image',
      affects: 'execution',
      visible: (p) => p.inputKind === 'image',
      help: 'Optional per-channel std, comma-separated. Blank = no normalization.',
    },

    // --- Text params ---
    {
      id: 'tokenizer',
      name: 'Tokenizer',
      type: {
        kind: 'select',
        options: [
          { label: 'Word', value: 'word' },
          { label: 'Character', value: 'char' },
          { label: 'BPE (subword)', value: 'bpe' },
        ],
      },
      defaultValue: 'word',
      group: 'Text',
      affects: 'execution',
      visible: (p) => p.inputKind === 'text',
      help: 'How text is split into tokens.',
    },
    {
      id: 'vocabSize',
      name: 'Vocab Size',
      type: { kind: 'number', min: 10, integer: true },
      defaultValue: 10000,
      group: 'Text',
      affects: 'execution',
      visible: (p) => p.inputKind === 'text',
      help: 'Number of distinct tokens to keep; the rest map to <unk>.',
    },
    {
      id: 'maxLen',
      name: 'Max Length',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 128,
      group: 'Text',
      affects: 'both',
      visible: (p) => p.inputKind === 'text' && p.task !== 'language_model',
      help: 'Sequence length for classification/regression. Longer texts are truncated, shorter padded.',
    },
    {
      id: 'seqLen',
      name: 'Context Length',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 128,
      group: 'Text',
      affects: 'both',
      visible: (p) => p.inputKind === 'text' && p.task === 'language_model',
      help: 'Number of tokens per training window for next-token language modeling.',
    },
  ],

  getPorts: () => [
    {
      id: 'out',
      name: 'Input',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
    {
      id: 'labels',
      name: 'Labels',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ properties: p }) => {
        const B = p.batchSize;
        let out: number[];
        let labels: number[];
        if (p.inputKind === 'image') {
          out = [B, p.channels, p.imageSize, p.imageSize];
          labels = [B];
        } else if (p.inputKind === 'text') {
          if (p.task === 'language_model') {
            out = [B, p.seqLen];
            labels = [B, p.seqLen]; // next-token targets
          } else {
            out = [B, p.maxLen];
            labels = [B];
          }
        } else {
          // vector: feature dim isn't known without loading — placeholder.
          out = [B];
          labels = [B];
        }
        return {
          outputs: { out, labels },
          metadata: {
            outputShape: out,
            datasetType: 'data.custom',
            shapes: [
              { label: 'Input', value: out },
              { label: 'Labels', value: labels },
              { label: 'Source', value: p.hfId || '(set HF dataset id)' },
              { label: 'Task', value: `${p.inputKind} · ${p.task}` },
            ],
          },
        };
      },
    },
  },
};
