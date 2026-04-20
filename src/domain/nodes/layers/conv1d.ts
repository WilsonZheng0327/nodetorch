import type { NodeDefinition } from '../../../core/nodedef';

export const conv1dNode: NodeDefinition = {
  type: 'ml.layers.conv1d',
  version: 1,
  displayName: 'Conv1d',
  description: '1D convolution (for sequences/signals)',
  category: ['ML', 'Layers', 'Convolution'],
  learnMore: 'The 1D version of Conv2d \u2014 slides filters along a sequence (text, audio, time series) instead of an image. Useful for detecting local patterns in sequential data without the overhead of recurrent networks.',

  getProperties: () => [
    { id: 'outChannels', name: 'Out Channels', type: { kind: 'number', min: 1, integer: true }, defaultValue: 64, affects: 'both' },
    { id: 'kernelSize', name: 'Kernel Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 3, affects: 'execution' },
    { id: 'stride', name: 'Stride', type: { kind: 'number', min: 1, integer: true }, defaultValue: 1, affects: 'execution' },
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
        if (!input) return { outputs: {} };

        const [B, C, L] = input;
        const { outChannels: OC, kernelSize: K, stride: S, padding: P } = properties;
        const outL = Math.floor((L + 2 * P - K) / S) + 1;

        if (outL <= 0) {
          return { outputs: {}, metadata: { error: `Invalid output length: ${outL} (kernel ${K} too large for length ${L})` } };
        }

        return {
          outputs: { out: [B, OC, outL] },
          metadata: {
            paramCount: OC * C * K + OC,
            paramBreakdown: `weights: ${OC}x${C}x${K} = ${OC * C * K} + bias: ${OC} = ${OC * C * K + OC}`,
            outputShape: [B, OC, outL],
            shapes: [
              { label: 'Output', value: [B, OC, outL] },
              { label: 'Weights', value: [OC, C, K] },
              { label: 'Bias', value: [OC] },
            ],
          },
        };
      },
    },
  },
};
