import type { NodeDefinition } from '../../../core/nodedef';

export const groupNormNode: NodeDefinition = {
  type: 'ml.layers.groupnorm',
  version: 1,
  displayName: 'GroupNorm',
  description: 'Group normalization — normalizes across groups of channels',
  category: ['ML', 'Layers', 'Normalization'],
  learnMore: 'Divides channels into groups and normalizes within each group. A middle ground between BatchNorm (normalize across batch) and InstanceNorm (normalize per channel). Works well with small batch sizes where BatchNorm is unstable.',

  getProperties: () => [
    { id: 'numGroups', name: 'Num Groups', type: { kind: 'number', min: 1, integer: true }, defaultValue: 8, affects: 'execution' },
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

        const C = input[1];
        const G = properties.numGroups;
        if (C % G !== 0) {
          return { outputs: {}, metadata: { error: `Channels ${C} not divisible by ${G} groups` } };
        }

        return {
          outputs: { out: input },
          metadata: {
            paramCount: C * 2,
            outputShape: input,
            shapes: [{ label: 'Output', value: input }],
          },
        };
      },
    },
  },
};
