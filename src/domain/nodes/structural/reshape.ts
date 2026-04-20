// Reshape — changes tensor shape without changing data.
// Target shape as comma-separated string. Use -1 for one inferred dimension.
// e.g., "32, -1" or "0, 16, 16" (0 = keep original dimension)

import type { NodeDefinition } from '../../../core/nodedef';

export const reshapeNode: NodeDefinition = {
  type: 'ml.structural.reshape',
  version: 1,
  displayName: 'Reshape',
  description: 'Reshape tensor (use -1 for inferred dim)',
  category: ['ML', 'Structural'],
  learnMore: 'Changes the shape of a tensor without changing its data. Use -1 for one dimension to let PyTorch infer it from the total number of elements. Common use: converting a flat vector back to spatial dimensions in decoder networks.',

  getProperties: () => [
    {
      id: 'targetShape',
      name: 'Target Shape',
      type: { kind: 'string' },
      defaultValue: '-1',
      affects: 'execution',
      help: 'Comma-separated dimensions. Use -1 for one inferred dim (batch size), 0 to keep original. E.g. "-1, 8, 7, 7" reshapes a flat vector back to [batch, 8, 7, 7].',
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

        const targetStr = properties.targetShape ?? '-1';
        const target = targetStr.split(',').map((s: string) => parseInt(s.trim()));

        // Replace 0s with original dimensions
        const resolved = target.map((v: number, i: number) => v === 0 ? (input[i] ?? v) : v);

        // Compute total elements
        const inputTotal = input.reduce((a: number, b: number) => a * b, 1);

        // Resolve -1
        const negIdx = resolved.indexOf(-1);
        if (negIdx !== -1) {
          const known = resolved.reduce((a: number, b: number) => b === -1 ? a : a * b, 1);
          if (known === 0 || inputTotal % known !== 0) {
            return { outputs: {}, metadata: { error: `Cannot reshape [${input}] to [${resolved}]` } };
          }
          resolved[negIdx] = inputTotal / known;
        }

        // Validate total elements match
        const outTotal = resolved.reduce((a: number, b: number) => a * b, 1);
        if (outTotal !== inputTotal) {
          return { outputs: {}, metadata: { error: `Element count mismatch: ${inputTotal} vs ${outTotal}` } };
        }

        return {
          outputs: { out: resolved },
          metadata: {
            outputShape: resolved,
            shapes: [{ label: 'Output', value: resolved }],
          },
        };
      },
    },
  },
};
