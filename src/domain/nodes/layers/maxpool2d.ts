import type { NodeDefinition } from '../../../core/nodedef';

export const maxPool2dNode: NodeDefinition = {
  type: 'ml.layers.maxpool2d',
  version: 1,
  displayName: 'MaxPool2d',
  description: '2D max pooling',
  category: ['ML', 'Layers', 'Pooling'],
  learnMore: 'Takes the maximum value in each pooling window, reducing spatial size while keeping the strongest features. A kernel of 2 with stride 2 halves the image dimensions. Provides translation invariance \u2014 the exact position of a feature matters less.',

  getProperties: () => [
    {
      id: 'kernelSize',
      name: 'Kernel Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 2,
      affects: 'execution',
      help: 'Size of the pooling window. 2 = takes max of each 2x2 region.',
    },
    {
      id: 'stride',
      name: 'Stride',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 2,
      affects: 'execution',
      help: 'Step size. Usually equals kernelSize. Stride 2 with kernel 2 halves spatial dimensions.',
    },
    {
      id: 'padding',
      name: 'Padding',
      type: { kind: 'number', min: 0, integer: true },
      defaultValue: 0,
      affects: 'execution',
      help: 'Zero-padding before pooling. Usually 0.',
    },
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
          return {
            outputs: {},
            metadata: { error: `Invalid output: ${outH}x${outW} (kernel ${K} too large for ${H}x${W} input)` },
          };
        }

        return {
          outputs: { out: [B, C, outH, outW] },
          metadata: {
            outputShape: [B, C, outH, outW],
            shapes: [{ label: 'Output', value: [B, C, outH, outW] }],
          },
        };
      },
    },
  },
};
