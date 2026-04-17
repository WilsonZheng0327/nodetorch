import type { NodeDefinition } from '../../../core/nodedef';

export const convTranspose2dNode: NodeDefinition = {
  type: 'ml.layers.conv_transpose2d',
  version: 1,
  displayName: 'ConvTranspose2d',
  description: 'Transposed 2D convolution (deconvolution)',
  category: ['ML', 'Layers', 'Convolution'],

  getProperties: () => [
    { id: 'outChannels', name: 'Out Channels', type: { kind: 'number', min: 1, integer: true }, defaultValue: 64, affects: 'both' },
    { id: 'kernelSize', name: 'Kernel Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 3, affects: 'execution' },
    { id: 'stride', name: 'Stride', type: { kind: 'number', min: 1, integer: true }, defaultValue: 1, affects: 'execution' },
    { id: 'padding', name: 'Padding', type: { kind: 'number', min: 0, integer: true }, defaultValue: 0, affects: 'execution' },
    { id: 'outputPadding', name: 'Output Padding', type: { kind: 'number', min: 0, integer: true }, defaultValue: 0, affects: 'execution' },
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

        const [B, C, H, W] = input;
        const { outChannels: OC, kernelSize: K, stride: S, padding: P, outputPadding: OP } = properties;
        const outH = (H - 1) * S - 2 * P + K + OP;
        const outW = (W - 1) * S - 2 * P + K + OP;

        if (outH <= 0 || outW <= 0) {
          return { outputs: {}, metadata: { error: `Invalid output: ${outH}x${outW}` } };
        }

        return {
          outputs: { out: [B, OC, outH, outW] },
          metadata: {
            paramCount: C * OC * K * K + OC,
            outputShape: [B, OC, outH, outW],
            shapes: [{ label: 'Output', value: [B, OC, outH, outW] }],
          },
        };
      },
    },
  },
};
