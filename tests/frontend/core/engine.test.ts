// Integration test for ExecutionEngine.execute, focused on the subgraph
// recursion path: a composite (subgraph.block) node must inject its inputs into
// the inner GraphInput sentinel, run the inner graph, and surface the inner
// GraphOutput as the block's own output.

import { describe, it, expect } from 'vitest';
import { createGraph, createNode, addNode, addEdge } from '../../../src/core/graph';
import { initDomain } from '../../../src/domain';

describe('ExecutionEngine — subgraph recursion (shape mode)', () => {
  it('propagates a shape through a composite block via its sentinels', async () => {
    const domain = initDomain();
    const lookup = (type: string, key: string) => domain.nodeRegistry.get(type)?.executors[key];

    // Inner graph: input → flatten → output.
    const inner = createGraph('inner', 'Inner');
    addNode(
      inner,
      createNode('si', 'subgraph.input', { x: 0, y: 0 }, { portCount: 1, portNames: 'in' }),
    );
    addNode(inner, createNode('fl', 'ml.layers.flatten', { x: 120, y: 0 }, {}));
    addNode(
      inner,
      createNode('so', 'subgraph.output', { x: 240, y: 0 }, { portCount: 1, portNames: 'out' }),
    );
    addEdge(inner, {
      id: 'ie1',
      source: { nodeId: 'si', portId: 'in' },
      target: { nodeId: 'fl', portId: 'in' },
    });
    addEdge(inner, {
      id: 'ie2',
      source: { nodeId: 'fl', portId: 'out' },
      target: { nodeId: 'so', portId: 'out' },
    });

    // Outer graph: mnist (provides a [B,1,28,28] tensor) → block.
    const outer = createGraph('main', 'Main');
    addNode(outer, createNode('mnist', 'data.mnist', { x: 0, y: 0 }, {}));
    const block = createNode('blk', 'subgraph.block', { x: 200, y: 0 }, { blockName: 'B' });
    block.subgraph = inner;
    addNode(outer, block);
    addEdge(outer, {
      id: 'e1',
      source: { nodeId: 'mnist', portId: 'out' },
      target: { nodeId: 'blk', portId: 'in' },
    });

    await domain.engine.execute(outer, 'shape', lookup);

    // The inner flatten actually ran (recursion happened).
    const flResult = inner.nodes.get('fl')!.lastResult;
    expect(flResult?.metadata?.outputShape).toBeDefined();

    // The block surfaced the inner output as its own, and flatten collapsed the
    // 4D image to 2D [B, features].
    const blockShape = outer.nodes.get('blk')!.lastResult?.metadata?.outputShape;
    expect(Array.isArray(blockShape)).toBe(true);
    expect((blockShape as number[]).length).toBe(2);
  });
});
