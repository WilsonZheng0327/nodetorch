// GraphInput sentinel — placed inside a subgraph to define its input ports.
// Each output port on this node becomes an input port on the parent SubGraph node.

import type { NodeDefinition } from '../../../core/nodedef';
import { SUBGRAPH_INPUT_TYPE } from '../../../core/graph';

export const graphInputNode: NodeDefinition = {
  type: SUBGRAPH_INPUT_TYPE,
  version: 1,
  displayName: 'Custom Block Input',
  description:
    'Data entering this custom block from outside. Place inside a Custom Block to define its input ports.',
  category: ['Custom Block'],
  color: '#f59e0b',

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
      defaultValue: 'in',
      affects: 'ports',
      help: 'Comma-separated labels for the ports, e.g. "x, y". Renaming is safe and only changes the label, not connections. Extra ports use default labels.',
    },
  ],

  // Port id is index-based and stable (never changes with a rename), so editing
  // the label leaves existing connections intact. The name is display-only.
  getPorts: (properties) => {
    const count = properties.portCount ?? 1;
    const names = String(properties.portNames ?? '')
      .split(',')
      .map((s: string) => s.trim());
    return Array.from({ length: count }, (_, i) => ({
      id: i === 0 ? 'in' : `port_${i}`,
      name: names[i] || (i === 0 ? 'in' : `in ${i}`),
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
