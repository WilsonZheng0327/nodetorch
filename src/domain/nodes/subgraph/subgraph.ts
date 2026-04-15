// SubGraph node — a node that contains an inner graph.
// Its ports are derived from GraphInput/GraphOutput sentinel nodes inside.
// Double-click to enter and edit the inner graph.
// Port resolution and recursive execution are handled by core/ports.ts and core/engine.ts.

import type { NodeDefinition } from '../../../core/nodedef';

export const subgraphNode: NodeDefinition = {
  type: 'subgraph.block',
  version: 1,
  displayName: 'Custom Block',
  description: 'A reusable block containing a subgraph',
  category: ['Block'],
  color: '#f97316',

  getProperties: () => [
    {
      id: 'blockName',
      name: 'Block Name',
      type: { kind: 'string' },
      defaultValue: 'Custom Block',
      affects: undefined,
    },
  ],

  // Ports are dynamic — derived from inner graph sentinels.
  // This default returns empty; getNodePorts() in core/ports.ts
  // handles the actual resolution by reading the inner graph.
  getPorts: () => [],

  executors: {
    shape: {
      // Shape execution is handled by the engine's subgraph recursion.
      // This fallback runs only if the node has no inner graph.
      execute: async () => {
        return { outputs: {}, metadata: { shapes: [{ label: 'Status', value: 'empty block' }] } };
      },
    },
  },
};
