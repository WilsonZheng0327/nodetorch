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
        const totalParams = 11689512;
        const trainableParams = properties.freeze ? 0 : totalParams;

        return {
          outputs: { out: outShape },
          metadata: {
            outputShape: outShape,
            paramCount: trainableParams,
            paramBreakdown: [
              `Total parameters: ${totalParams.toLocaleString()}`,
              `Trainable: ${trainableParams.toLocaleString()} (${properties.freeze ? 'frozen' : 'all unfrozen'})`,
              `Architecture: 4 residual blocks (BasicBlock \u00d7 [2, 2, 2, 2])`,
              `Pretrained on: ImageNet (1.2M images, 1000 classes, ~69.8% top-1 acc)`,
              `Mode: ${properties.mode === 'logits' ? 'Full model \u2192 1000-class logits' : 'Feature extractor \u2192 512-dim vector'}`,
              `Input: auto-resized to 224\u00d7224`,
            ].join('\n'),
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Input', value: 'auto-resized to 224\u00d7224' },
              { label: 'Pretrained', value: 'ImageNet (69.8% top-1)' },
              { label: 'Total params', value: '11.7M' },
              { label: 'Trainable', value: properties.freeze ? '0 (frozen)' : '11.7M' },
            ],
          },
        };
      },
    },
  },
};
