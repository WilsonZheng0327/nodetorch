// GRU — Gated Recurrent Unit recurrent layer.
// Input: [B, seq_len, input_size]
// Outputs:
//   out: [B, seq_len, hidden_size * num_directions]
//   hidden: [num_layers * num_directions, B, hidden_size]

import type { NodeDefinition } from '../../../core/nodedef';

export const gruNode: NodeDefinition = {
  type: 'ml.layers.gru',
  version: 1,
  displayName: 'GRU',
  description: 'Gated Recurrent Unit layer',
  category: ['ML', 'Layers', 'Recurrent'],

  getProperties: () => [
    { id: 'hiddenSize', name: 'Hidden Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 128, affects: 'execution' },
    { id: 'numLayers', name: 'Num Layers', type: { kind: 'number', min: 1, integer: true }, defaultValue: 1, affects: 'execution' },
    { id: 'bidirectional', name: 'Bidirectional', type: { kind: 'boolean' }, defaultValue: false, affects: 'execution' },
    { id: 'dropout', name: 'Dropout', type: { kind: 'number', min: 0, max: 1, step: 0.1 }, defaultValue: 0, affects: 'execution' },
  ],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
    { id: 'hidden', name: 'Hidden', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input) return { outputs: {} };

        if (input.length !== 3) {
          return { outputs: {}, metadata: { error: `Input should be [B, seq_len, input_size], got [${input}]` } };
        }

        const [B, seqLen, inputSize] = input;
        const { hiddenSize: H, numLayers: L, bidirectional } = properties;
        const D = bidirectional ? 2 : 1;

        const outShape = [B, seqLen, H * D];
        const hiddenShape = [L * D, B, H];

        // Params: 3 gates × (input_size × hidden + hidden × hidden + 2 biases) × layers × directions
        const firstLayerParams = 3 * (inputSize * H + H * H + 2 * H);
        const otherLayerParams = L > 1 ? (L - 1) * 3 * (H * D * H + H * H + 2 * H) : 0;
        const paramCount = (firstLayerParams + otherLayerParams) * D;

        return {
          outputs: { out: outShape, hidden: hiddenShape },
          metadata: {
            outputShape: outShape,
            paramCount,
            shapes: [
              { label: 'Output', value: outShape },
              { label: 'Hidden', value: hiddenShape },
            ],
          },
        };
      },
    },
  },
};
