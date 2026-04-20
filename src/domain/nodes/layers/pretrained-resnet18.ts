import type { NodeDefinition } from '../../../core/nodedef';

// ResNet-18 pretrained on ImageNet. Good for transfer learning.
// Note: expects RGB input. Inputs not 224×224 are resized internally.
// ImageNet normalization is assumed — if your data uses dataset-specific mean/std,
// results may be suboptimal.

export const pretrainedResnet18Node: NodeDefinition = {
  type: 'ml.layers.pretrained_resnet18',
  version: 1,
  displayName: 'ResNet18 (pretrained)',
  description: 'Pretrained ResNet-18 from torchvision. Good for transfer learning.',
  category: ['ML', 'Pretrained'],
  learnMore: 'A ResNet-18 model pre-trained on 1.2 million ImageNet images. Already knows how to detect edges, textures, shapes, and objects. In "features" mode, it outputs a 512-dimensional feature vector you can classify with a simple Linear layer \u2014 this is transfer learning. Much faster than training from scratch.',

  getProperties: () => [
    {
      id: 'freeze',
      name: 'Freeze Weights',
      type: { kind: 'boolean' },
      defaultValue: true,
      affects: 'execution',
    },
    {
      id: 'mode',
      name: 'Output Mode',
      type: {
        kind: 'select',
        options: [
          { label: 'Features [B, 512]', value: 'features' },
          { label: 'Logits [B, 1000]', value: 'logits' },
        ],
      },
      defaultValue: 'features',
      affects: 'both',
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
        if (input.length !== 4 || input[1] !== 3) {
          return { outputs: {}, metadata: { error: 'Expected [B, 3, H, W] RGB input' } };
        }

        const B = input[0];
        const outShape = properties.mode === 'logits' ? [B, 1000] : [B, 512];
        // resnet18 total trainable params: ~11.7M. 0 if frozen.
        const paramCount = properties.freeze ? 0 : 11689512;

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            paramCount,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Input → 224×224', value: 'auto-resized' },
            ],
          },
        };
      },
    },
  },
};
