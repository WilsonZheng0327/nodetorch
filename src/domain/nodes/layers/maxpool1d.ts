import type { NodeDefinition } from '../../../core/nodedef';

export const maxPool1dNode: NodeDefinition = {
  type: 'ml.layers.maxpool1d',
  version: 1,
  displayName: 'MaxPool1d',
  description: '1D max pooling',
  category: ['ML', 'Layers', 'Pooling'],

  getProperties: () => [
    { id: 'kernelSize', name: 'Kernel Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 2, affects: 'execution' },
    { id: 'stride', name: 'Stride', type: { kind: 'number', min: 1, integer: true }, defaultValue: 2, affects: 'execution' },
    { id: 'padding', name: 'Padding', type: { kind: 'number', min: 0, integer: true }, defaultValue: 0, affects: 'execution' },
  ],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input || input.length !== 3) {
          return input ? { outputs: {}, metadata: { error: 'MaxPool1d expects [B, C, L]' } } : { outputs: {} };
        }

        const [B, C, L] = input;
        const { kernelSize: K, stride: S, padding: P } = properties;
        const outL = Math.floor((L + 2 * P - K) / S) + 1;

        if (outL <= 0) {
          return { outputs: {}, metadata: { error: `Invalid output length: ${outL}` } };
        }

        return {
          outputs: { out: [B, C, outL] },
          metadata: { outputShape: [B, C, outL], shapes: [{ label: 'Output', value: [B, C, outL] }] },
        };
      },
    },
  },
};
