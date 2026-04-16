// GraphInput sentinel — placed inside a subgraph to define its input ports.
// Each output port on this node becomes an input port on the parent SubGraph node.

import type { NodeDefinition } from '../../../core/nodedef';

export const graphInputNode: NodeDefinition = {
  type: 'subgraph.input',
  version: 1,
  displayName: 'Block Input',
  description: 'Data entering this custom block from outside. Place inside a Custom Block to define its input ports.',
  category: ['Block'],
  color: '#f59e0b',

  getProperties: () => [
    {
      id: 'portCount',
      name: 'Port Count',
      type: { kind: 'number', min: 1, max: 8, integer: true },
      defaultValue: 1,
      affects: 'ports',
    },
  ],

  getPorts: (properties) => {
    const count = properties.portCount ?? 1;
    const names = (properties.portNames ?? 'in').split(',').map((s: string) => s.trim());
    return Array.from({ length: count }, (_, i) => ({
      id: names[i] || `port_${i}`,
      name: names[i] || `Port ${i}`,
      direction: 'output' as const,
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    }));
  },

  executors: {
    shape: {
      // Shape data is injected by the parent subgraph's executor
      execute: async ({ inputs }) => {
        const outputs: Record<string, any> = {};
        for (const [key, value] of Object.entries(inputs)) {
          outputs[key] = value;
        }
        return { outputs };
      },
    },
  },
};
