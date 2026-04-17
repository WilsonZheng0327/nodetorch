import type { NodeDefinition } from '../../../core/nodedef';

export const upsampleNode: NodeDefinition = {
  type: 'ml.layers.upsample',
  version: 1,
  displayName: 'Upsample',
  description: 'Upsamples input by scale factor (nearest/bilinear)',
  category: ['ML', 'Layers', 'Pooling'],

  getProperties: () => [
    { id: 'scaleFactor', name: 'Scale Factor', type: { kind: 'number', min: 1, integer: true }, defaultValue: 2, affects: 'execution' },
    {
      id: 'mode', name: 'Mode',
      type: { kind: 'select', options: [{ label: 'Nearest', value: 'nearest' }, { label: 'Bilinear', value: 'bilinear' }] },
      defaultValue: 'nearest', affects: 'execution',
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

        const S = properties.scaleFactor;
        if (input.length === 4) {
          const [B, C, H, W] = input;
          const out = [B, C, H * S, W * S];
          return { outputs: { out }, metadata: { outputShape: out, shapes: [{ label: 'Output', value: out }] } };
        }
        if (input.length === 3) {
          const [B, C, L] = input;
          const out = [B, C, L * S];
          return { outputs: { out }, metadata: { outputShape: out, shapes: [{ label: 'Output', value: out }] } };
        }
        return { outputs: {}, metadata: { error: 'Upsample requires 3D or 4D input' } };
      },
    },
  },
};
