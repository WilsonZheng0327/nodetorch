import { describe, it, expect } from 'vitest';
import {
  createGraph,
  createNode,
  createEdge,
  addNode,
  addEdge,
  removeNode,
  topologicalSort,
  markDirty,
  setProperty,
} from '../../../src/core/graph';

// ---------------------------------------------------------------------------
// createGraph
// ---------------------------------------------------------------------------
describe('createGraph', () => {
  it('creates a graph with the given id and name', () => {
    const g = createGraph('g1', 'My Graph');
    expect(g.id).toBe('g1');
    expect(g.name).toBe('My Graph');
  });

  it('starts with empty nodes, edges, and metadata', () => {
    const g = createGraph('g2', 'Empty');
    expect(g.nodes.size).toBe(0);
    expect(g.edges).toEqual([]);
    expect(g.metadata).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// createNode
// ---------------------------------------------------------------------------
describe('createNode', () => {
  it('creates a node with the given id and type', () => {
    const n = createNode('n1', 'ml.layers.conv2d');
    expect(n.id).toBe('n1');
    expect(n.type).toBe('ml.layers.conv2d');
  });

  it('uses default position (0,0) and empty properties when omitted', () => {
    const n = createNode('n1', 'ml.layers.conv2d');
    expect(n.position).toEqual({ x: 0, y: 0 });
    expect(n.properties).toEqual({});
  });

  it('accepts a custom position and properties', () => {
    const n = createNode('n1', 'ml.layers.conv2d', { x: 100, y: 200 }, { kernelSize: 3 });
    expect(n.position).toEqual({ x: 100, y: 200 });
    expect(n.properties).toEqual({ kernelSize: 3 });
  });

  it('starts dirty with null state and no errors', () => {
    const n = createNode('n1', 'test');
    expect(n.dirty).toBe(true);
    expect(n.state).toBeNull();
    expect(n.errors).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// createEdge
// ---------------------------------------------------------------------------
describe('createEdge', () => {
  it('creates an edge with source and target port references', () => {
    const e = createEdge('e1', 'n1', 'out', 'n2', 'in');
    expect(e.id).toBe('e1');
    expect(e.source).toEqual({ nodeId: 'n1', portId: 'out' });
    expect(e.target).toEqual({ nodeId: 'n2', portId: 'in' });
  });

  it('preserves exact port ids', () => {
    const e = createEdge('e2', 'conv1', 'output', 'relu1', 'input');
    expect(e.source.portId).toBe('output');
    expect(e.target.portId).toBe('input');
  });
});

// ---------------------------------------------------------------------------
// addNode
// ---------------------------------------------------------------------------
describe('addNode', () => {
  it('adds a node to the graph', () => {
    const g = createGraph('g', 'test');
    const n = createNode('n1', 'test');
    addNode(g, n);
    expect(g.nodes.size).toBe(1);
    expect(g.nodes.get('n1')).toBe(n);
  });

  it('allows adding multiple nodes', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addNode(g, createNode('n3', 'c'));
    expect(g.nodes.size).toBe(3);
  });

  it('throws if a node with the same id already exists', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'test'));
    expect(() => addNode(g, createNode('n1', 'test'))).toThrow('already exists');
  });
});

// ---------------------------------------------------------------------------
// addEdge
// ---------------------------------------------------------------------------
describe('addEdge', () => {
  it('adds an edge between two existing nodes', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    const e = createEdge('e1', 'n1', 'out', 'n2', 'in');
    addEdge(g, e);
    expect(g.edges).toHaveLength(1);
    expect(g.edges[0]).toBe(e);
  });

  it('throws if the source node does not exist', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n2', 'b'));
    expect(() => addEdge(g, createEdge('e1', 'missing', 'out', 'n2', 'in'))).toThrow(
      'Source node "missing" not found',
    );
  });

  it('throws if the target node does not exist', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    expect(() => addEdge(g, createEdge('e1', 'n1', 'out', 'missing', 'in'))).toThrow(
      'Target node "missing" not found',
    );
  });

  it('throws on duplicate edge (same source+target port pair)', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    expect(() => addEdge(g, createEdge('e2', 'n1', 'out', 'n2', 'in'))).toThrow('already exists');
  });

  it('allows edges to different ports on the same node pair', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addEdge(g, createEdge('e1', 'n1', 'out1', 'n2', 'in1'));
    addEdge(g, createEdge('e2', 'n1', 'out2', 'n2', 'in2'));
    expect(g.edges).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// removeNode
// ---------------------------------------------------------------------------
describe('removeNode', () => {
  it('removes the node from the graph', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    removeNode(g, 'n1');
    expect(g.nodes.size).toBe(1);
    expect(g.nodes.has('n1')).toBe(false);
  });

  it('removes all edges connected to the removed node', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addNode(g, createNode('n3', 'c'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    addEdge(g, createEdge('e2', 'n2', 'out', 'n3', 'in'));
    removeNode(g, 'n2');
    expect(g.edges).toHaveLength(0);
  });

  it('preserves edges not connected to the removed node', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addNode(g, createNode('n3', 'c'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    addEdge(g, createEdge('e2', 'n1', 'out', 'n3', 'in'));
    removeNode(g, 'n2');
    expect(g.edges).toHaveLength(1);
    expect(g.edges[0].target.nodeId).toBe('n3');
  });

  it('throws if the node does not exist', () => {
    const g = createGraph('g', 'test');
    expect(() => removeNode(g, 'missing')).toThrow('not found');
  });
});

// ---------------------------------------------------------------------------
// topologicalSort
// ---------------------------------------------------------------------------
describe('topologicalSort', () => {
  it('returns a single node for a graph with one node', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    expect(topologicalSort(g)).toEqual(['n1']);
  });

  it('returns nodes in dependency order (upstream first)', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addNode(g, createNode('n3', 'c'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    addEdge(g, createEdge('e2', 'n2', 'out', 'n3', 'in'));
    const sorted = topologicalSort(g);
    expect(sorted.indexOf('n1')).toBeLessThan(sorted.indexOf('n2'));
    expect(sorted.indexOf('n2')).toBeLessThan(sorted.indexOf('n3'));
  });

  it('handles a diamond-shaped graph correctly', () => {
    // n1 -> n2, n1 -> n3, n2 -> n4, n3 -> n4
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    addNode(g, createNode('n3', 'c'));
    addNode(g, createNode('n4', 'd'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    addEdge(g, createEdge('e2', 'n1', 'out', 'n3', 'in'));
    addEdge(g, createEdge('e3', 'n2', 'out', 'n4', 'in'));
    addEdge(g, createEdge('e4', 'n3', 'out', 'n4', 'in'));
    const sorted = topologicalSort(g);
    expect(sorted).toHaveLength(4);
    expect(sorted.indexOf('n1')).toBeLessThan(sorted.indexOf('n2'));
    expect(sorted.indexOf('n1')).toBeLessThan(sorted.indexOf('n3'));
    expect(sorted.indexOf('n2')).toBeLessThan(sorted.indexOf('n4'));
    expect(sorted.indexOf('n3')).toBeLessThan(sorted.indexOf('n4'));
  });

  it('returns all nodes for a disconnected graph (no edges)', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('a', 'x'));
    addNode(g, createNode('b', 'y'));
    addNode(g, createNode('c', 'z'));
    const sorted = topologicalSort(g);
    expect(sorted).toHaveLength(3);
    expect(sorted).toContain('a');
    expect(sorted).toContain('b');
    expect(sorted).toContain('c');
  });

  it('throws if the graph has a cycle', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    addNode(g, createNode('n2', 'b'));
    // Manually push edges to create a cycle (addEdge prevents self-loops but not cycles)
    g.edges.push(createEdge('e1', 'n1', 'out', 'n2', 'in'));
    g.edges.push(createEdge('e2', 'n2', 'out', 'n1', 'in'));
    expect(() => topologicalSort(g)).toThrow('cycle');
  });
});

// ---------------------------------------------------------------------------
// markDirty
// ---------------------------------------------------------------------------
describe('markDirty', () => {
  it('marks the target node as dirty', () => {
    const g = createGraph('g', 'test');
    const n = createNode('n1', 'a');
    n.dirty = false;
    addNode(g, n);
    markDirty(g, 'n1');
    expect(n.dirty).toBe(true);
  });

  it('marks all downstream nodes as dirty', () => {
    const g = createGraph('g', 'test');
    const n1 = createNode('n1', 'a');
    const n2 = createNode('n2', 'b');
    const n3 = createNode('n3', 'c');
    n1.dirty = false;
    n2.dirty = false;
    n3.dirty = false;
    addNode(g, n1);
    addNode(g, n2);
    addNode(g, n3);
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    addEdge(g, createEdge('e2', 'n2', 'out', 'n3', 'in'));
    markDirty(g, 'n1');
    expect(n1.dirty).toBe(true);
    expect(n2.dirty).toBe(true);
    expect(n3.dirty).toBe(true);
  });

  it('does not mark upstream nodes as dirty', () => {
    const g = createGraph('g', 'test');
    const n1 = createNode('n1', 'a');
    const n2 = createNode('n2', 'b');
    const n3 = createNode('n3', 'c');
    n1.dirty = false;
    n2.dirty = false;
    n3.dirty = false;
    addNode(g, n1);
    addNode(g, n2);
    addNode(g, n3);
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    addEdge(g, createEdge('e2', 'n2', 'out', 'n3', 'in'));
    markDirty(g, 'n2');
    expect(n1.dirty).toBe(false);
    expect(n2.dirty).toBe(true);
    expect(n3.dirty).toBe(true);
  });

  it('silently does nothing if the node id does not exist', () => {
    const g = createGraph('g', 'test');
    expect(() => markDirty(g, 'nonexistent')).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// setProperty
// ---------------------------------------------------------------------------
describe('setProperty', () => {
  it('sets the property value on the node', () => {
    const g = createGraph('g', 'test');
    const n = createNode('n1', 'a', { x: 0, y: 0 }, { kernelSize: 3 });
    addNode(g, n);
    setProperty(g, 'n1', 'kernelSize', 5);
    expect(n.properties.kernelSize).toBe(5);
  });

  it('can add a new property that did not exist before', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'a'));
    setProperty(g, 'n1', 'newProp', 'hello');
    expect(g.nodes.get('n1')!.properties.newProp).toBe('hello');
  });

  it('marks the node and downstream nodes as dirty', () => {
    const g = createGraph('g', 'test');
    const n1 = createNode('n1', 'a');
    const n2 = createNode('n2', 'b');
    n1.dirty = false;
    n2.dirty = false;
    addNode(g, n1);
    addNode(g, n2);
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    setProperty(g, 'n1', 'x', 42);
    expect(n1.dirty).toBe(true);
    expect(n2.dirty).toBe(true);
  });

  it('throws if the node does not exist', () => {
    const g = createGraph('g', 'test');
    expect(() => setProperty(g, 'missing', 'x', 1)).toThrow('not found');
  });
});
