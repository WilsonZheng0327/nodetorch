import type { NodeDefinition } from '../../../core/nodedef';

export const convTranspose2dNode: NodeDefinition = {
  type: 'ml.layers.conv_transpose2d',
  version: 1,
  displayName: 'ConvTranspose2d',
  description: 'Transposed 2D convolution (deconvolution)',
  category: ['ML', 'Layers', 'Convolution'],
  learnMore: 'The "reverse" of Conv2d \u2014 increases spatial resolution instead of decreasing it. Used in decoder networks (autoencoders, GANs, segmentation) to upsample feature maps back to image size. Also called "deconvolution" though that term is technically incorrect.',

  getProperties: () => [
    { id: 'outChannels', name: 'Out Channels', type: { kind: 'number', min: 1, integer: true }, defaultValue: 64, affects: 'both', help: 'Number of output channels.' },
    { id: 'kernelSize', name: 'Kernel Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 3, affects: 'execution', help: 'Size of the upsampling filter.' },
    { id: 'stride', name: 'Stride', type: { kind: 'number', min: 1, integer: true }, defaultValue: 1, affects: 'execution', help: 'Upsampling factor. Stride 2 doubles spatial dimensions.' },
    { id: 'padding', name: 'Padding', type: { kind: 'number', min: 0, integer: true }, defaultValue: 0, affects: 'execution', help: 'Controls output size. With kernel=4, stride=2, padding=1: output = 2x input size.' },
    { id: 'outputPadding', name: 'Output Padding', type: { kind: 'number', min: 0, integer: true }, defaultValue: 0, affects: 'execution', help: 'Extra padding on one side to resolve ambiguous output sizes. Usually 0.' },
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
