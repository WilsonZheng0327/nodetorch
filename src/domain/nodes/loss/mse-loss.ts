import type { NodeDefinition } from '../../../core/nodedef';

export const mseLossNode: NodeDefinition = {
  type: 'ml.loss.mse',
  version: 1,
  displayName: 'MSELoss',
  description: 'Mean Squared Error loss (regression)',
  category: ['ML', 'Loss'],
  color: '#ef4444',
  learnMore: 'Measures the average squared difference between predictions and targets. Used for regression tasks and reconstruction (autoencoders, diffusion). Penalizes large errors more than small ones because of the squaring.',

  getProperties: () => [],

  getPorts: () => [
    { id: 'predictions', name: 'Predictions', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'labels', name: 'Targets', direction: 'input', dataType: 'tensor', allowMultiple: false, optional: false },
    { id: 'out', name: 'Loss', direction: 'output', dataType: 'scalar', allowMultiple: true, optional: false },
  ],

  executors: {
    shape: {
      execute: async ({ inputs }) => {
        const predictions = inputs.predictions;
        const labels = inputs.labels;
        if (!predictions || !labels) return { outputs: {} };

        // Shapes should match
        if (JSON.stringify(predictions) !== JSON.stringify(labels)) {
          return {
            outputs: {},
            metadata: { error: `Shape mismatch: predictions ${JSON.stringify(predictions)} vs targets ${JSON.stringify(labels)}` },
          };
        }

        return {
          outputs: { out: [] },
          metadata: {
            outputShape: ['scalar'],
            shapes: [{ label: 'Output', value: 'scalar' }],
          },
        };
      },
    },
  },
};
