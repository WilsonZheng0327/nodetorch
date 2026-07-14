// Tests for src/core/ports.ts — the single place that answers "what ports does
// this node instance have?". Regular nodes read their NodeDefinition; subgraph
// (composite block) nodes derive external ports from their inner I/O sentinels.

import { describe, it, expect } from 'vitest';
import {
  createGraph,
  createNode,
  addNode,
  SUBGRAPH_INPUT_TYPE,
  SUBGRAPH_OUTPUT_TYPE,
} from '../../../src/core/graph';
import { getNodePorts, getSubgraphPorts } from '../../../src/core/ports';
import { initDomain } from '../../../src/domain';

const domain = initDomain();
const registry = domain.nodeRegistry;

describe('getNodePorts — regular nodes', () => {
  it('reads input and output ports straight from the NodeDefinition', () => {
    const relu = createNode('r', 'ml.activations.relu', { x: 0, y: 0 }, {});
    const ports = getNodePorts(relu, registry);
    const ins = ports.filter((p) => p.direction === 'input').map((p) => p.id);
    const outs = ports.filter((p) => p.direction === 'output').map((p) => p.id);
    expect(ins).toContain('in');
    expect(outs).toContain('out');
  });

  it('returns [] for an unregistered node type', () => {
    const bogus = createNode('b', 'does.not.exist', { x: 0, y: 0 }, {});
    expect(getNodePorts(bogus, registry)).toEqual([]);
  });
});

describe('getSubgraphPorts — derived from inner sentinels', () => {
  function makeInner(inputPorts: number, outputPorts: number) {
    const inner = createGraph('inner', 'Inner');
    addNode(
      inner,
      createNode('si', SUBGRAPH_INPUT_TYPE, { x: 0, y: 0 }, { portCount: inputPorts, portNames: 'a, b, c' }),
    );
    addNode(
      inner,
      createNode('so', SUBGRAPH_OUTPUT_TYPE, { x: 200, y: 0 }, { portCount: outputPorts, portNames: 'y, z' }),
    );
    return inner;
  }

  it('turns GraphInput output ports into block INPUT ports (single-connection)', () => {
    const ports = getSubgraphPorts(makeInner(2, 1), registry);
    const ins = ports.filter((p) => p.direction === 'input');
    expect(ins).toHaveLength(2);
    // Inputs on a block accept one edge each.
    expect(ins.every((p) => p.allowMultiple === false)).toBe(true);
  });

  it('turns GraphOutput input ports into block OUTPUT ports (fan-out allowed)', () => {
    const ports = getSubgraphPorts(makeInner(2, 1), registry);
    const outs = ports.filter((p) => p.direction === 'output');
    expect(outs).toHaveLength(1);
    expect(outs.every((p) => p.allowMultiple === true)).toBe(true);
  });

  it('reflects the sentinels\' port counts (2 in / 3 out)', () => {
    const ports = getSubgraphPorts(makeInner(2, 3), registry);
    expect(ports.filter((p) => p.direction === 'input')).toHaveLength(2);
    expect(ports.filter((p) => p.direction === 'output')).toHaveLength(3);
  });
});

describe('getNodePorts — subgraph nodes route through getSubgraphPorts', () => {
  it('a block node exposes its inner sentinels as its own ports', () => {
    const inner = createGraph('inner', 'Inner');
    addNode(inner, createNode('si', SUBGRAPH_INPUT_TYPE, { x: 0, y: 0 }, { portCount: 1, portNames: 'in' }));
    addNode(inner, createNode('so', SUBGRAPH_OUTPUT_TYPE, { x: 200, y: 0 }, { portCount: 1, portNames: 'out' }));

    const block = createNode('blk', 'subgraph.block', { x: 0, y: 0 }, { blockName: 'B' });
    block.subgraph = inner;

    const ports = getNodePorts(block, registry);
    expect(ports.filter((p) => p.direction === 'input')).toHaveLength(1);
    expect(ports.filter((p) => p.direction === 'output')).toHaveLength(1);
  });
});
