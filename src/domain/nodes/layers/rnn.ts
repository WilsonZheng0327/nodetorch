import type { NodeDefinition } from '../../../core/nodedef';

export const rnnNode: NodeDefinition = {
  type: 'ml.layers.rnn',
  version: 1,
  displayName: 'RNN',
  description: 'Basic recurrent neural network',
  category: ['ML', 'Layers', 'Recurrent'],
  learnMore: 'The simplest recurrent layer \u2014 processes sequences by passing a hidden state from one timestep to the next. Struggles with long sequences due to vanishing gradients. In practice, LSTM or GRU are almost always preferred.',

  getProperties: () => [
    { id: 'hiddenSize', name: 'Hidden Size', type: { kind: 'number', min: 1, integer: true }, defaultValue: 128, affects: 'both' },
    { id: 'numLayers', name: 'Num Layers', type: { kind: 'number', min: 1, max: 8, integer: true }, defaultValue: 1, affects: 'both' },
    { id: 'bidirectional', name: 'Bidirectional', type: { kind: 'boolean' }, defaultValue: false, affects: 'both' },
    { id: 'nonlinearity', name: 'Nonlinearity', type: { kind: 'select', options: [{ label: 'tanh', value: 'tanh' }, { label: 'relu', value: 'relu' }] }, defaultValue: 'tanh', affects: 'execution' },
  ],

  getPorts: () => [
    { id: 'in', name: 'Input', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: false },
    { id: 'hidden', name: 'Hidden', direction: 'output', dataType: 'tensor', allowMultiple: true, optional: true },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;
        if (!input || input.length !== 3) {
          return input ? { outputs: {}, metadata: { error: 'RNN expects [batch, seq_len, input_size]' } } : { outputs: {} };
        }

        const [B, seqLen, inputSize] = input;
        const { hiddenSize: H, numLayers: L, bidirectional } = properties;
        const D = bidirectional ? 2 : 1;

        const outShape = [B, seqLen, H * D];
        const hiddenShape = [L * D, B, H];

        const firstLayerParams = D * (inputSize * H + H * H + 2 * H);
        const otherLayerParams = (L - 1) * D * (H * D * H + H * H + 2 * H);

        return {
          outputs: { out: outShape, hidden: hiddenShape },
          metadata: {
            paramCount: firstLayerParams + otherLayerParams,
            outputShape: outShape,
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
