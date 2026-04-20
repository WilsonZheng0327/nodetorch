// Conv2d layer node.
// Shape executor does the math: output = floor((input + 2*padding - kernel) / stride) + 1

import type { NodeDefinition } from '../../../core/nodedef';

export const conv2dNode: NodeDefinition = {
  type: 'ml.layers.conv2d',
  version: 1,
  displayName: 'Conv2d',
  description: '2D convolution layer',
  category: ['ML', 'Layers', 'Convolution'],
  learnMore: 'Slides small filters across the image to detect patterns like edges, textures, and shapes. Early layers learn simple patterns (edges), deeper layers learn complex ones (eyes, wheels). The key building block of all image models. Use padding=1 with kernel=3 to keep the spatial size unchanged.',

  getProperties: () => [
    {
      id: 'outChannels',
      name: 'Out Channels',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 64,
      affects: 'both',
      help: 'Number of filters (output feature maps). More filters = more patterns detected. Common: 32, 64, 128, 256.',
    },
    {
      id: 'kernelSize',
      name: 'Kernel Size',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 3,
      affects: 'execution',
      help: 'Size of the sliding window (e.g. 3 = 3x3 filter). Larger kernels see more context but have more parameters. 3 is the most common choice.',
    },
    {
      id: 'stride',
      name: 'Stride',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 1,
      affects: 'execution',
      help: 'Step size of the sliding window. Stride 2 halves the spatial dimensions (downsampling). Stride 1 keeps the same size (with appropriate padding).',
    },
    {
      id: 'padding',
      name: 'Padding',
      type: { kind: 'number', min: 0, integer: true },
      defaultValue: 0,
      affects: 'execution',
      help: 'Zero-padding added around the input. Use padding=1 with kernel=3 to keep spatial size unchanged. Formula: same size when padding = (kernel-1)/2.',
    },
  ],

  getPorts: () => [
    {
      id: 'in',
      name: 'Input',
      direction: 'input',
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'out',
      name: 'Output',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) {
          return { outputs: {} };
        }

        const [B, C, H, W] = input;
        const { outChannels: OC, kernelSize: K, stride: S, padding: P } = properties;
        const outH = Math.floor((H + 2 * P - K) / S) + 1;
        const outW = Math.floor((W + 2 * P - K) / S) + 1;

        // Validate output dimensions
        if (outH <= 0 || outW <= 0) {
          return {
            outputs: {},
            metadata: {
              error: `Invalid output: ${outH}x${outW} (kernel ${K} too large for ${H}x${W} input)`,
            },
          };
        }

        return {
          outputs: { out: [B, OC, outH, outW] },
          metadata: {
            paramCount: OC * C * K * K + OC,
            paramBreakdown: `weights: ${OC}x${C}x${K}x${K} = ${OC * C * K * K}  +  bias: ${OC}  =  ${OC * C * K * K + OC}`,
            outputShape: [B, OC, outH, outW],
            shapes: [
              { label: 'Output', value: [B, OC, outH, outW] },
              { label: 'Weights', value: [OC, C, K, K] },
              { label: 'Bias', value: [OC] },
            ],
          },
        };
      },
    },
  },
};
