// Tests for the agent graph-tool executor (src/ui/chat/graphTools.ts).
// Uses a fake GraphToolApi backed by a real Graph so we exercise the tool
// dispatch, validation, and result strings without the full useGraph hook.

import { describe, it, expect } from 'vitest';

import { initDomain } from '../../../src/domain';
import { createGraph, createNode, addNode } from '../../../src/core/graph';
import { executeGraphTool, type GraphToolApi } from '../../../src/ui/chat/graphTools';

const domain = initDomain();

function makeApi() {
  const g = createGraph('main', 'Main');
  addNode(g, createNode('c1', 'ml.layers.conv2d', { x: 0, y: 0 }, { outChannels: 32 }));

  const api: GraphToolApi = {
    currentGraph: g,
    updateProperty: async (id, k, v) => {
      const n = g.nodes.get(id);
      if (n) (n.properties as Record<string, unknown>)[k] = v;
    },
    addNode: async (type, pos) => {
      const id = `${type}-new`;
      addNode(g, createNode(id, type, pos, {}));
      return id;
    },
    connect: async () => {},
    removeNode: async (id) => {
      g.nodes.delete(id);
    },
    // fake validation: reject only the sentinel 'bad' target port
    isValidConnection: (c) => c.targetHandle !== 'bad',
  };
  return { api, g };
}

describe('executeGraphTool', () => {
  it('set_node_property applies, and errors on missing node / invalid key', async () => {
    const { api, g } = makeApi();
    expect(await executeGraphTool(api, domain, 'set_node_property', { nodeId: 'c1', key: 'outChannels', value: 64 })).toMatch(/^ok/);
    expect(g.nodes.get('c1')!.properties.outChannels).toBe(64);
    expect(await executeGraphTool(api, domain, 'set_node_property', { nodeId: 'nope', key: 'x', value: 1 })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'set_node_property', { nodeId: 'c1', key: 'bogus', value: 1 })).toMatch(/^error/);
  });

  it('add_node ok for a valid type, error for unknown, and returns the new id', async () => {
    const { api, g } = makeApi();
    const res = await executeGraphTool(api, domain, 'add_node', { type: 'ml.activations.relu' });
    expect(res).toMatch(/^ok: added ml\.activations\.relu/);
    expect([...g.nodes.values()].some((n) => n.type === 'ml.activations.relu')).toBe(true);
    expect(await executeGraphTool(api, domain, 'add_node', { type: 'not.a.type' })).toMatch(/^error/);
  });

  it('connect validates and errors on missing node / invalid port', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    expect(await executeGraphTool(api, domain, 'connect', { sourceId: 'c1', sourcePort: 'out', targetId: 'r1', targetPort: 'in' })).toMatch(/^ok/);
    expect(await executeGraphTool(api, domain, 'connect', { sourceId: 'c1', sourcePort: 'out', targetId: 'nope', targetPort: 'in' })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'connect', { sourceId: 'c1', sourcePort: 'out', targetId: 'r1', targetPort: 'bad' })).toMatch(/invalid connection/);
  });

  it('remove_node ok then error when already gone', async () => {
    const { api, g } = makeApi();
    expect(await executeGraphTool(api, domain, 'remove_node', { nodeId: 'c1' })).toMatch(/^ok/);
    expect(g.nodes.has('c1')).toBe(false);
    expect(await executeGraphTool(api, domain, 'remove_node', { nodeId: 'c1' })).toMatch(/^error/);
  });

  it('reports unknown tools', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'frobnicate', {})).toMatch(/unknown tool/);
  });
});
