// Tests for the agent graph-tool executor (src/ui/chat/graphTools.ts).
// Uses a fake GraphToolApi backed by a real Graph so we exercise the tool
// dispatch, validation, and result strings without the full useGraph hook.

import { describe, it, expect } from 'vitest';

import { initDomain } from '../../../src/domain';
import { createGraph, createNode, addNode, createEdge, addEdge } from '../../../src/core/graph';
import { executeGraphTool, type GraphToolApi } from '../../../src/ui/chat/graphTools';

const domain = initDomain();

function makeApi() {
  const g = createGraph('main', 'Main');
  addNode(g, createNode('c1', 'ml.layers.conv2d', { x: 0, y: 0 }, { outChannels: 32 }));
  const flags = { organized: false, blockAdded: '', entered: '', exited: 0, saved: '' };

  const api: GraphToolApi = {
    getCurrentGraph: () => g,
    updateProperty: async (id, k, v) => {
      const n = g.nodes.get(id);
      if (n) (n.properties as Record<string, unknown>)[k] = v;
    },
    addNode: async (type, pos, requestedId) => {
      const id = requestedId && !g.nodes.has(requestedId) ? requestedId : `${type}-new`;
      addNode(g, createNode(id, type, pos, {}));
      return id;
    },
    connect: async () => {},
    removeNode: async (id) => {
      g.nodes.delete(id);
    },
    removeEdge: async (s, sp, t, tp) => {
      const i = g.edges.findIndex(
        (e) => e.source.nodeId === s && e.source.portId === sp && e.target.nodeId === t && e.target.portId === tp,
      );
      if (i < 0) return false;
      g.edges.splice(i, 1);
      return true;
    },
    clearGraph: () => {
      g.nodes.clear();
      g.edges = [];
    },
    organizeGraph: () => {
      flags.organized = true;
    },
    addBlockFromTemplate: async (filename) => {
      flags.blockAdded = filename;
    },
    enterSubgraph: (id) => {
      flags.entered = id;
    },
    exitSubgraph: () => {
      flags.exited += 1;
    },
    saveBlock: async (id) => {
      flags.saved = id;
    },
    // fake validation: reject only the sentinel 'bad' target port
    isValidConnection: (c) => c.targetHandle !== 'bad',
  };
  return { api, g, flags };
}

describe('executeGraphTool', () => {
  it('set_node_property applies, and errors on missing node / invalid key', async () => {
    const { api, g } = makeApi();
    expect(await executeGraphTool(api, domain, 'set_node_property', { nodeId: 'c1', key: 'outChannels', value: 64 })).toMatch(/^ok/);
    expect(g.nodes.get('c1')!.properties.outChannels).toBe(64);
    expect(await executeGraphTool(api, domain, 'set_node_property', { nodeId: 'nope', key: 'x', value: 1 })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'set_node_property', { nodeId: 'c1', key: 'bogus', value: 1 })).toMatch(/^error/);
  });

  it('add_node ok for a valid type, error for unknown, honors a requested id', async () => {
    const { api, g } = makeApi();
    expect(await executeGraphTool(api, domain, 'add_node', { type: 'ml.activations.relu' })).toMatch(/^ok: added ml\.activations\.relu/);
    expect([...g.nodes.values()].some((n) => n.type === 'ml.activations.relu')).toBe(true);
    expect(await executeGraphTool(api, domain, 'add_node', { type: 'not.a.type' })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'add_node', { type: 'ml.activations.relu', id: 'relu1' })).toMatch(/as "relu1"/);
    expect(g.nodes.has('relu1')).toBe(true);
  });

  it('connect validates and errors on missing node / invalid port', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    expect(await executeGraphTool(api, domain, 'connect', { sourceId: 'c1', sourcePort: 'out', targetId: 'r1', targetPort: 'in' })).toMatch(/^ok/);
    expect(await executeGraphTool(api, domain, 'connect', { sourceId: 'c1', sourcePort: 'out', targetId: 'nope', targetPort: 'in' })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'connect', { sourceId: 'c1', sourcePort: 'out', targetId: 'r1', targetPort: 'bad' })).toMatch(/invalid connection/);
  });

  it('remove_node / remove_edge', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    addEdge(g, createEdge('e1', 'c1', 'out', 'r1', 'in'));
    expect(await executeGraphTool(api, domain, 'remove_edge', { sourceId: 'c1', sourcePort: 'out', targetId: 'r1', targetPort: 'in' })).toMatch(/^ok/);
    expect(g.edges.length).toBe(0);
    expect(await executeGraphTool(api, domain, 'remove_edge', { sourceId: 'c1', sourcePort: 'out', targetId: 'r1', targetPort: 'in' })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'remove_node', { nodeId: 'c1' })).toMatch(/^ok/);
    expect(g.nodes.has('c1')).toBe(false);
  });

  it('clear_graph and organize_layout', async () => {
    const { api, g, flags } = makeApi();
    expect(await executeGraphTool(api, domain, 'clear_graph', {})).toMatch(/^ok/);
    expect(g.nodes.size).toBe(0);
    expect(await executeGraphTool(api, domain, 'organize_layout', {})).toMatch(/^ok/);
    expect(flags.organized).toBe(true);
  });

  it('add_block passes the filename through', async () => {
    const { api, flags } = makeApi();
    expect(await executeGraphTool(api, domain, 'add_block', { filename: 'resblock.json' })).toMatch(/^ok/);
    expect(flags.blockAdded).toBe('resblock.json');
  });

  it('enter_block / exit_block / save_block on a subgraph.block node', async () => {
    const { api, g, flags } = makeApi();
    const block = createNode('blk1', 'subgraph.block', { x: 2, y: 0 }, { blockName: 'MyBlock' });
    block.subgraph = createGraph('blk1-inner', 'MyBlock');
    addNode(g, block);

    // enter/save require a real subgraph.block; a plain node is rejected.
    expect(await executeGraphTool(api, domain, 'enter_block', { nodeId: 'c1' })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'enter_block', { nodeId: 'blk1' })).toMatch(/^ok/);
    expect(flags.entered).toBe('blk1');

    expect(await executeGraphTool(api, domain, 'exit_block', {})).toMatch(/^ok/);
    expect(flags.exited).toBe(1);

    expect(await executeGraphTool(api, domain, 'save_block', { nodeId: 'blk1', name: 'Renamed' })).toMatch(/^ok/);
    expect(flags.saved).toBe('blk1');
    expect(g.nodes.get('blk1')!.properties.blockName).toBe('Renamed');
    expect(await executeGraphTool(api, domain, 'save_block', { nodeId: 'c1' })).toMatch(/^error/);
  });

  it('get_graph and get_node report state', async () => {
    const { api } = makeApi();
    const summary = await executeGraphTool(api, domain, 'get_graph', {});
    expect(summary).toMatch(/c1 \(ml\.layers\.conv2d\)/);
    const node = await executeGraphTool(api, domain, 'get_node', { nodeId: 'c1' });
    expect(node).toMatch(/c1 \(ml\.layers\.conv2d\)/);
    expect(node).toMatch(/outChannels/);
    expect(await executeGraphTool(api, domain, 'get_node', { nodeId: 'nope' })).toMatch(/^error/);
  });

  it('validate returns a forward/training report', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'validate', { mode: 'forward' })).toMatch(/forward/);
    expect(await executeGraphTool(api, domain, 'validate', { mode: 'training' })).toMatch(/training/);
  });

  it('reports unknown tools', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'frobnicate', {})).toMatch(/unknown tool/);
  });
});
