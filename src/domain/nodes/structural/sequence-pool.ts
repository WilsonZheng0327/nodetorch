import type { NodeDefinition } from '../../../core/nodedef';

export const sequencePoolNode: NodeDefinition = {
  type: 'ml.structural.sequence_pool',
  version: 1,
  displayName: 'SeqPool',
  description: 'Pool over sequence dimension: [B, seq, H] → [B, H]',
  category: ['ML', 'Structural'],

  getProperties: () => [
    {
      id: 'mode',
      name: 'Mode',
      type: {
        kind: 'select',
        options: [
          { label: 'Last timestep', value: 'last' },
          { label: 'Mean', value: 'mean' },
          { label: 'Max', value: 'max' },
        ],
      },
      defaultValue: 'last',
      affects: 'execution',
      help: "How to reduce the sequence to a single vector. 'last' takes the final timestep (good for LSTM). 'mean' averages all timesteps. 'max' takes element-wise maximum.",
    },
  ],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };

        if (input.length !== 3) {
          return { outputs: {}, metadata: { error: `Expected [B, seq, H], got ${input.length}D` } };
        }

        const [B, , H] = input;
        const out = [B, H];

        return {
          outputs: { out },
          metadata: {
            outputShape: out,
            shapes: [{ label: 'Output', value: out }],
          },
        };
      },
    },
  },
};
