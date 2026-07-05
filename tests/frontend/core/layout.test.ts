// Tests for the pure auto-layout algorithm (longest-path layering).

import { describe, it, expect } from 'vitest';
import { createGraph, createNode, addNode, addEdge } from '../../../src/core/graph';
import { computeLayout } from '../../../src/core/layout';

function wire(g: ReturnType<typeof createGraph>, from: string, to: string) {
  addEdge(g, {
    id: `${from}-${to}`,
    source: { nodeId: from, portId: 'out' },
    target: { nodeId: to, portId: 'in' },
  });
}

describe('computeLayout', () => {
  it('returns an empty map for an empty graph', () => {
    expect(computeLayout(createGraph('g', 'G')).size).toBe(0);
  });

  it('places a chain in strictly increasing columns (longest path)', () => {
    const g = createGraph('g', 'G');
    addNode(g, createNode('a', 'ml.activations.relu', { x: 0, y: 0 }, {}));
    addNode(g, createNode('b', 'ml.activations.relu', { x: 0, y: 0 }, {}));
    addNode(g, createNode('c', 'ml.activations.relu', { x: 0, y: 0 }, {}));
    wire(g, 'a', 'b');
    wire(g, 'b', 'c');

    const pos = computeLayout(g, { colGap: 100 });
    expect(pos.get('a')!.x).toBe(0);
    expect(pos.get('b')!.x).toBe(100);
    expect(pos.get('c')!.x).toBe(200);
  });

  it('uses the LONGEST path when a node has a shortcut edge', () => {
    // a→b→c and a→c : c must land past b, not adjacent to a.
    const g = createGraph('g', 'G');
    for (const id of ['a', 'b', 'c']) {
      addNode(g, createNode(id, 'ml.activations.relu', { x: 0, y: 0 }, {}));
    }
    wire(g, 'a', 'b');
    wire(g, 'b', 'c');
    wire(g, 'a', 'c');

    const pos = computeLayout(g, { colGap: 100 });
    expect(pos.get('c')!.x).toBe(200); // depth 2, not depth 1
  });

  it('puts disconnected nodes in column 0', () => {
    const g = createGraph('g', 'G');
    addNode(g, createNode('a', 'ml.activations.relu', { x: 0, y: 0 }, {}));
    addNode(g, createNode('lonely', 'ml.activations.relu', { x: 0, y: 0 }, {}));
    wire(g, 'a', 'a'); // self-loop should not crash; treat as its own column

    const pos = computeLayout(g, { colGap: 100 });
    expect(pos.get('lonely')!.x).toBe(0);
  });

  it('preserves top-to-bottom order within a column and centers it', () => {
    const g = createGraph('g', 'G');
    addNode(g, createNode('top', 'ml.activations.relu', { x: 0, y: 10 }, {}));
    addNode(g, createNode('bot', 'ml.activations.relu', { x: 0, y: 90 }, {}));

    const pos = computeLayout(g, { rowGap: 40 });
    // Both are roots → same column; 'top' (smaller y) stays above 'bot'.
    expect(pos.get('top')!.x).toBe(pos.get('bot')!.x);
    expect(pos.get('top')!.y).toBeLessThan(pos.get('bot')!.y);
    // Centered around 0: two nodes at ±20.
    expect(pos.get('top')!.y).toBe(-20);
    expect(pos.get('bot')!.y).toBe(20);
  });
});
