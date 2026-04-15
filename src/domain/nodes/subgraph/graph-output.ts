// GraphOutput sentinel — placed inside a subgraph to define its output ports.
// Each input port on this node becomes an output port on the parent SubGraph node.

import type { NodeDefinition } from '../../../core/nodedef';

export const graphOutputNode: NodeDefinition = {
  type: 'subgraph.output',
  version: 1,
  displayName: 'Block Output',
  description: 'Data leaving this custom block to outside. Place inside a Custom Block to define its output ports.',
  category: ['Block'],
  color: '#8b5cf6',

  getProperties: () => [
    {
      id: 'portCount',
      name: 'Port Count',
      type: { kind: 'number', min: 1, max: 8, integer: true },
      defaultValue: 1,
      affects: 'ports',
    },
    {
      id: 'portNames',
      name: 'Port Names',
      type: { kind: 'string' },
      defaultValue: 'out',
      affects: 'ports',
    },
  ],

  getPorts: (properties) => {
    const count = properties.portCount ?? 1;
    const names = (properties.portNames ?? 'out').split(',').map((s: string) => s.trim());
    return Array.from({ length: count }, (_, i) => ({
      id: names[i] || `port_${i}`,
      name: names[i] || `Port ${i}`,
      direction: 'input' as const,
      dataType: 'tensor',
      allowMultiple: false,
      optional: false,
    }));
  },

  executors: {
    shape: {
      // Collect inputs and pass them up to the parent subgraph
      execute: async ({ inputs }) => {
        return {
          outputs: inputs,
          metadata: {
            shapes: Object.entries(inputs).map(([key, value]) => ({
              label: key,
              value: Array.isArray(value) ? value : String(value),
            })),
          },
        };
      },
    },
  },
};
