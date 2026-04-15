import type { NodeDefinition } from '../../../core/nodedef';

export const avgPool2dNode: NodeDefinition = {
  type: 'ml.layers.avgpool2d',
  version: 1,
  displayName: 'AvgPool2d',
  description: '2D average pooling',
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
        if (!input) return { outputs: {} };

        const [B, C, H, W] = input;
        const { kernelSize: K, stride: S, padding: P } = properties;
        const outH = Math.floor((H + 2 * P - K) / S) + 1;
        const outW = Math.floor((W + 2 * P - K) / S) + 1;

        if (outH <= 0 || outW <= 0) {
          return { outputs: {}, metadata: { error: `Invalid output: ${outH}x${outW}` } };
        }

        return {
          outputs: { out: [B, C, outH, outW] },
          metadata: { outputShape: [B, C, outH, outW], shapes: [{ label: 'Output', value: [B, C, outH, outW] }] },
        };
      },
    },
  },
};
