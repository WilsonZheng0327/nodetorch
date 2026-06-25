// Tests for the SerializedGraph format: serialize→deserialize round-trips
// (including nested subgraphs) and the pre-load structural validator.

import { describe, it, expect } from 'vitest';

import { createGraph, createNode, addNode, addEdge, type Graph } from '../../../src/core/graph';
import {
  serializeGraph,
  deserializeGraph,
  validateSerializedGraph,
  SERIALIZED_GRAPH_VERSION,
} from '../../../src/core/serialization';

/** A small graph: two nodes wired out→in, plus one composite node with an inner graph. */
function makeGraph(): Graph {
  const g = createGraph('main', 'Main');
  addNode(g, createNode('a', 'ml.layers.conv2d', { x: 0, y: 0 }, { outChannels: 32 }));
  addNode(g, createNode('b', 'ml.activations.relu', { x: 200, y: 0 }, {}));
  addEdge(g, { id: 'e1', source: { nodeId: 'a', portId: 'out' }, target: { nodeId: 'b', portId: 'in' } });

  // A composite node whose subgraph has its own nodes + edge.
  const block = createNode('blk', 'subgraph.block', { x: 400, y: 0 }, { blockName: 'My Block' });
  const inner = createGraph('inner', 'Inner');
  addNode(inner, createNode('i1', 'subgraph.input', { x: 0, y: 0 }, {}));
  addNode(inner, createNode('i2', 'ml.layers.linear', { x: 100, y: 0 }, { outFeatures: 10 }));
  addEdge(inner, { id: 'ie1', source: { nodeId: 'i1', portId: 'out' }, target: { nodeId: 'i2', portId: 'in' } });
  block.subgraph = inner;
  addNode(g, block);
  return g;
}

describe('serialization round-trip', () => {
  it('reproduces nodes, edges, and nested subgraphs structurally', () => {
    const restored = deserializeGraph(serializeGraph(makeGraph()));

    expect([...restored.nodes.keys()].sort()).toEqual(['a', 'b', 'blk']);
    const a = restored.nodes.get('a')!;
    expect(a.type).toBe('ml.layers.conv2d');
    expect(a.position).toEqual({ x: 0, y: 0 });
    expect(a.properties.outChannels).toBe(32);

    expect(restored.edges).toHaveLength(1);
    expect(restored.edges[0]).toMatchObject({
      id: 'e1',
      source: { nodeId: 'a', portId: 'out' },
      target: { nodeId: 'b', portId: 'in' },
    });

    // Nested subgraph survives with its nodes + edge intact.
    const inner = restored.nodes.get('blk')!.subgraph!;
    expect(inner).toBeDefined();
    expect([...inner.nodes.keys()].sort()).toEqual(['i1', 'i2']);
    expect(inner.edges).toHaveLength(1);
    expect(inner.edges[0].source.nodeId).toBe('i1');
  });

  it('a re-serialized restored graph is byte-identical to the first serialization (idempotent)', () => {
    const once = serializeGraph(makeGraph());
    const twice = serializeGraph(deserializeGraph(once));
    expect(twice).toEqual(once);
    expect(once.version).toBe(SERIALIZED_GRAPH_VERSION);
  });
});

describe('validateSerializedGraph', () => {
  const known = (t: string) => t === 'ml.layers.conv2d' || t === 'ml.activations.relu';

  it('passes a well-formed graph', () => {
    const data = serializeGraph(makeGraph());
    // No type predicate → structural-only, should be clean.
    expect(validateSerializedGraph(data)).toEqual([]);
  });

  it('rejects non-objects and a wrong version', () => {
    expect(validateSerializedGraph(null)).toHaveLength(1);
    expect(validateSerializedGraph({ version: '0.9', graph: { id: 'm', name: 'm', nodes: [], edges: [] } }))
      .toEqual([expect.stringMatching(/version/i)]);
  });

  it('flags missing nodes list', () => {
    expect(validateSerializedGraph({ version: '1.0' })).toEqual([expect.stringMatching(/nodes/i)]);
  });

  it('flags an edge that references a missing node', () => {
    const data = {
      version: '1.0',
      graph: {
        id: 'm', name: 'm',
        nodes: [{ id: 'a', type: 'ml.activations.relu', position: { x: 0, y: 0 }, properties: {} }],
        edges: [{ id: 'e', source: { nodeId: 'a', portId: 'out' }, target: { nodeId: 'ghost', portId: 'in' } }],
      },
    };
    expect(validateSerializedGraph(data)).toEqual([expect.stringMatching(/missing node "ghost"/)]);
  });

  it('flags unknown node types when a type predicate is supplied', () => {
    const data = {
      version: '1.0',
      graph: {
        id: 'm', name: 'm',
        nodes: [{ id: 'x', type: 'ml.layers.quantum_flux', position: { x: 0, y: 0 }, properties: {} }],
        edges: [],
      },
    };
    expect(validateSerializedGraph(data, known)).toEqual([expect.stringMatching(/unknown node type/i)]);
  });

  it('flags duplicate node ids and recurses into subgraphs', () => {
    const data = {
      version: '1.0',
      graph: {
        id: 'm', name: 'm',
        nodes: [
          { id: 'dup', type: 'ml.activations.relu', position: { x: 0, y: 0 }, properties: {} },
          { id: 'dup', type: 'ml.activations.relu', position: { x: 0, y: 0 }, properties: {} },
          {
            id: 'blk', type: 'subgraph.block', position: { x: 0, y: 0 }, properties: {},
            subgraph: {
              id: 'i', name: 'i',
              nodes: [{ id: 'q', type: 'ml.activations.relu', position: { x: 0, y: 0 }, properties: {} }],
              edges: [{ id: 'e', source: { nodeId: 'q', portId: 'out' }, target: { nodeId: 'missing', portId: 'in' } }],
            },
          },
        ],
        edges: [],
      },
    };
    const issues = validateSerializedGraph(data);
    expect(issues.some((m) => /duplicate node id "dup"/.test(m))).toBe(true);
    // The subgraph's dangling edge is reported with a nested path.
    expect(issues.some((m) => /blk.*missing node "missing"/.test(m))).toBe(true);
  });
});
