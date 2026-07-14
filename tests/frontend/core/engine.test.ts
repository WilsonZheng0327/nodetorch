// Integration test for ExecutionEngine.execute, focused on the subgraph
// recursion path: a composite (subgraph.block) node must inject its inputs into
// the inner GraphInput sentinel, run the inner graph, and surface the inner
// GraphOutput as the block's own output.

import { describe, it, expect } from 'vitest';
import { type Graph, createGraph, createNode, addNode, addEdge } from '../../../src/core/graph';
import { ExecutionEngine } from '../../../src/core/engine';
import { initDomain } from '../../../src/domain';
import { buildSubgraphMetadata } from '../../../src/domain/subgraph-metadata';

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

  it('aggregates the inner layers\' params onto the block via the domain metadata builder', async () => {
    const domain = initDomain();
    const lookup = (type: string, key: string) => domain.nodeRegistry.get(type)?.executors[key];

    // Inner: input → flatten → linear → output. The linear layer has real params.
    const inner = createGraph('inner', 'Inner');
    addNode(inner, createNode('si', 'subgraph.input', { x: 0, y: 0 }, { portCount: 1, portNames: 'in' }));
    addNode(inner, createNode('fl', 'ml.layers.flatten', { x: 120, y: 0 }, {}));
    addNode(inner, createNode('ln', 'ml.layers.linear', { x: 240, y: 0 }, { outFeatures: 10 }));
    addNode(inner, createNode('so', 'subgraph.output', { x: 360, y: 0 }, { portCount: 1, portNames: 'out' }));
    addEdge(inner, { id: 'a', source: { nodeId: 'si', portId: 'in' }, target: { nodeId: 'fl', portId: 'in' } });
    addEdge(inner, { id: 'b', source: { nodeId: 'fl', portId: 'out' }, target: { nodeId: 'ln', portId: 'in' } });
    addEdge(inner, { id: 'c', source: { nodeId: 'ln', portId: 'out' }, target: { nodeId: 'so', portId: 'out' } });

    const outer = createGraph('main', 'Main');
    addNode(outer, createNode('mnist', 'data.mnist', { x: 0, y: 0 }, {}));
    const block = createNode('blk', 'subgraph.block', { x: 200, y: 0 }, { blockName: 'B' });
    block.subgraph = inner;
    addNode(outer, block);
    addEdge(outer, { id: 'e1', source: { nodeId: 'mnist', portId: 'out' }, target: { nodeId: 'blk', portId: 'in' } });

    await domain.engine.execute(outer, 'shape', lookup);

    const meta = outer.nodes.get('blk')!.lastResult?.metadata;
    // 784*10 + 10 = 7850 params, all from the one linear layer.
    expect(meta?.paramCount).toBe(7850);
    expect(meta?.paramBreakdown).toContain('linear');
    expect(meta?.paramBreakdown).toContain('7,850');
  });

  it('without a registered builder, the engine emits only generic block metadata (Layer 3 stays ML-agnostic)', async () => {
    // Fresh engine + registry, but DO NOT call setSubgraphMetadataBuilder.
    const domain = initDomain();
    const bareEngine = new ExecutionEngine();
    // Register the shape mode so the engine can run, but leave the builder unset.
    bareEngine.registerMode({
      id: 'shape',
      label: 'Shape',
      propagation: 'eager',
      caching: true,
      executorKey: 'shape',
    });
    const lookup = (type: string, key: string) => domain.nodeRegistry.get(type)?.executors[key];

    const inner = createGraph('inner', 'Inner');
    addNode(inner, createNode('si', 'subgraph.input', { x: 0, y: 0 }, { portCount: 1, portNames: 'in' }));
    addNode(inner, createNode('fl', 'ml.layers.flatten', { x: 120, y: 0 }, {}));
    addNode(inner, createNode('so', 'subgraph.output', { x: 240, y: 0 }, { portCount: 1, portNames: 'out' }));
    addEdge(inner, { id: 'a', source: { nodeId: 'si', portId: 'in' }, target: { nodeId: 'fl', portId: 'in' } });
    addEdge(inner, { id: 'b', source: { nodeId: 'fl', portId: 'out' }, target: { nodeId: 'so', portId: 'out' } });

    const outer = createGraph('main', 'Main');
    addNode(outer, createNode('mnist', 'data.mnist', { x: 0, y: 0 }, {}));
    const block = createNode('blk', 'subgraph.block', { x: 200, y: 0 }, { blockName: 'B' });
    block.subgraph = inner;
    addNode(outer, block);
    addEdge(outer, { id: 'e1', source: { nodeId: 'mnist', portId: 'out' }, target: { nodeId: 'blk', portId: 'in' } });

    await bareEngine.execute(outer, 'shape', lookup);

    const meta = outer.nodes.get('blk')!.lastResult?.metadata;
    // Generic metadata is present…
    expect(meta?.outputShape).toBeDefined();
    // …but the ML/presentation param breakdown is NOT (no builder registered).
    expect(meta?.paramBreakdown).toBeUndefined();
  });
});

describe('buildSubgraphMetadata — Layer 5 param aggregation', () => {
  function fakeSubgraph(parts: { type: string; paramCount?: number }[]): Graph {
    const nodes = new Map(
      parts.map((p, i) => [
        `n${i}`,
        { type: p.type, lastResult: { outputs: {}, metadata: { paramCount: p.paramCount } } },
      ]),
    );
    return { nodes } as unknown as Graph;
  }

  it('sums inner param counts and formats a per-layer breakdown', () => {
    const meta = buildSubgraphMetadata(
      fakeSubgraph([
        { type: 'ml.layers.conv2d', paramCount: 576 },
        { type: 'ml.layers.linear', paramCount: 128 },
        { type: 'ml.activations.relu' }, // no params — omitted from breakdown
      ]),
      { outputShape: [1, 10], shapes: [] },
    );
    expect(meta.paramCount).toBe(704);
    expect(meta.paramBreakdown).toBe('conv2d: 576 + linear: 128 = 704');
    // Base metadata is preserved (builder augments, doesn't replace).
    expect(meta.outputShape).toEqual([1, 10]);
  });

  it('leaves param fields undefined when no inner node has params', () => {
    const meta = buildSubgraphMetadata(
      fakeSubgraph([{ type: 'ml.activations.relu' }]),
      { outputShape: [1, 3], shapes: [] },
    );
    expect(meta.paramCount).toBeUndefined();
    expect(meta.paramBreakdown).toBeUndefined();
    expect(meta.outputShape).toEqual([1, 3]);
  });
});
