import type { NodeDefinition } from '../../../core/nodedef';

export const adaptiveAvgPool2dNode: NodeDefinition = {
  type: 'ml.layers.adaptive_avgpool2d',
  version: 1,
  displayName: 'AdaptiveAvgPool2d',
  description: 'Adaptive average pooling (specify output size)',
  category: ['ML', 'Layers', 'Pooling'],
  learnMore: 'Pools to a fixed output size regardless of input size. Commonly used as the last pooling layer to produce a consistent size before the classifier, no matter what the input resolution is.',

  getProperties: () => [
    { id: 'outputH', name: 'Output Height', type: { kind: 'number', min: 1, integer: true }, defaultValue: 1, affects: 'execution' },
    { id: 'outputW', name: 'Output Width', type: { kind: 'number', min: 1, integer: true }, defaultValue: 1, affects: 'execution' },
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

        const [B, C] = input;
        const outShape = [B, C, properties.outputH, properties.outputW];

        return {
          outputs: { out: outShape },
          metadata: { outputShape: outShape, shapes: [{ label: 'Output', value: outShape }] },
        };
      },
    },
  },
};
