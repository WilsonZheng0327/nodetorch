import { describe, it, expect } from 'vitest';
import {
  createGraph,
  createNode,
  createEdge,
  addNode,
  addEdge,
} from '../../../src/core/graph';
import { validateForward, validateTraining } from '../../../src/core/validation';
import { NodeRegistry } from '../../../src/core/nodedef';
import type { NodeDefinition, PortDefinition } from '../../../src/core/nodedef';

// ---------------------------------------------------------------------------
// Helper: create a minimal NodeDefinition
// ---------------------------------------------------------------------------
function makeDef(
  type: string,
  displayName: string,
  ports: PortDefinition[],
): NodeDefinition {
  return {
    type,
    version: 1,
    displayName,
    description: '',
    category: [],
    getProperties: () => [],
    getPorts: () => ports,
    executors: {},
  };
}

/** Shorthand for an input port. */
function inputPort(id: string, name: string, optional = false): PortDefinition {
  return { id, name, direction: 'input', dataType: 'tensor', allowMultiple: false, optional };
}

/** Shorthand for an output port. */
function outputPort(id: string, name: string): PortDefinition {
  return { id, name, direction: 'output', dataType: 'tensor', allowMultiple: false, optional: false };
}

// ---------------------------------------------------------------------------
// Build a registry with the node types used in tests
// ---------------------------------------------------------------------------
function buildRegistry(): NodeRegistry {
  const registry = new NodeRegistry();

  registry.register(makeDef('data.mnist', 'MNIST', [
    outputPort('images', 'Images'),
    outputPort('labels', 'Labels'),
  ]));

  registry.register(makeDef('data.cifar10', 'CIFAR-10', [
    outputPort('images', 'Images'),
    outputPort('labels', 'Labels'),
  ]));

  registry.register(makeDef('ml.layers.conv2d', 'Conv2d', [
    inputPort('in', 'Input'),
    outputPort('out', 'Output'),
  ]));

  registry.register(makeDef('ml.layers.linear', 'Linear', [
    inputPort('in', 'Input'),
    outputPort('out', 'Output'),
  ]));

  registry.register(makeDef('ml.layers.relu', 'ReLU', [
    inputPort('in', 'Input'),
    outputPort('out', 'Output'),
  ]));

  registry.register(makeDef('ml.loss.cross_entropy', 'CrossEntropyLoss', [
    inputPort('predictions', 'Predictions'),
    inputPort('labels', 'Labels'),
    outputPort('loss', 'Loss'),
  ]));

  registry.register(makeDef('ml.optimizers.sgd', 'SGD', [
    inputPort('loss', 'Loss'),
  ]));

  registry.register(makeDef('ml.optimizers.adam', 'Adam', [
    inputPort('loss', 'Loss'),
  ]));

  return registry;
}

// ---------------------------------------------------------------------------
// validateForward
// ---------------------------------------------------------------------------
describe('validateForward', () => {
  it('returns an error for an empty graph', () => {
    const g = createGraph('g', 'test');
    const errors = validateForward(g, buildRegistry());
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toMatch(/no nodes/i);
  });

  it('returns an error when there are nodes but no edges', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'ml.layers.conv2d'));
    addNode(g, createNode('n2', 'ml.layers.relu'));
    const errors = validateForward(g, buildRegistry());
    expect(errors).toHaveLength(1);
    expect(errors[0].message).toMatch(/no connections/i);
  });

  it('returns an error when a required input port is not connected', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'ml.layers.conv2d'));
    addNode(g, createNode('n2', 'ml.layers.relu'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    // conv2d's "in" port is required but nothing connects to it
    const errors = validateForward(g, buildRegistry());
    expect(errors.length).toBeGreaterThanOrEqual(1);
    const convError = errors.find((e) => e.nodeId === 'n1');
    expect(convError).toBeDefined();
    expect(convError!.message).toMatch(/not connected/i);
  });

  it('reports unknown node types', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'nonexistent.type'));
    addNode(g, createNode('n2', 'ml.layers.conv2d'));
    addEdge(g, createEdge('e1', 'n1', 'out', 'n2', 'in'));
    const errors = validateForward(g, buildRegistry());
    const unknownErr = errors.find((e) => e.nodeId === 'n1');
    expect(unknownErr).toBeDefined();
    expect(unknownErr!.message).toMatch(/unknown node type/i);
  });

  it('returns no errors for a valid simple graph', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('conv', 'ml.layers.conv2d'));
    addEdge(g, createEdge('e1', 'data', 'images', 'conv', 'in'));
    const errors = validateForward(g, buildRegistry());
    expect(errors).toHaveLength(0);
  });

  it('does not flag optional ports that lack connections', () => {
    // Create a node type with an optional input
    const registry = buildRegistry();
    registry.register(makeDef('custom.optional', 'OptionalInput', [
      inputPort('required', 'Required', false),
      inputPort('extra', 'Extra', true),
      outputPort('out', 'Output'),
    ]));

    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('opt', 'custom.optional'));
    addEdge(g, createEdge('e1', 'data', 'images', 'opt', 'required'));
    const errors = validateForward(g, registry);
    expect(errors).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// validateTraining
// ---------------------------------------------------------------------------
describe('validateTraining', () => {
  /**
   * Helper: build a fully-connected valid training graph.
   * data(mnist) -> linear -> loss(cross_entropy) -> optimizer(sgd)
   *           \---labels--> loss
   * All required ports are connected so forward validation passes.
   */
  function buildValidTrainingGraph() {
    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addNode(g, createNode('opt', 'ml.optimizers.sgd'));
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));
    addEdge(g, createEdge('e2', 'linear', 'out', 'loss', 'predictions'));
    addEdge(g, createEdge('e3', 'data', 'labels', 'loss', 'labels'));
    addEdge(g, createEdge('e4', 'loss', 'loss', 'opt', 'loss'));
    return g;
  }

  it('returns forward errors first (empty graph)', () => {
    const g = createGraph('g', 'test');
    const errors = validateTraining(g, buildRegistry());
    expect(errors.length).toBeGreaterThanOrEqual(1);
    expect(errors[0].message).toMatch(/no nodes/i);
  });

  it('returns forward errors first (no edges)', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('n1', 'data.mnist'));
    addNode(g, createNode('n2', 'ml.layers.linear'));
    const errors = validateTraining(g, buildRegistry());
    expect(errors.length).toBeGreaterThanOrEqual(1);
    expect(errors[0].message).toMatch(/no connections/i);
  });

  it('errors when no data node is present', () => {
    // Use a registry with a source node that has no required inputs
    const registry = buildRegistry();
    registry.register(makeDef('custom.source', 'Source', [
      outputPort('out', 'Output'),
      outputPort('labels', 'Labels'),
    ]));

    const g = createGraph('g', 'test');
    addNode(g, createNode('src', 'custom.source'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addNode(g, createNode('opt', 'ml.optimizers.sgd'));
    addEdge(g, createEdge('e1', 'src', 'out', 'linear', 'in'));
    addEdge(g, createEdge('e2', 'linear', 'out', 'loss', 'predictions'));
    addEdge(g, createEdge('e3', 'src', 'labels', 'loss', 'labels'));
    addEdge(g, createEdge('e4', 'loss', 'loss', 'opt', 'loss'));

    const errors = validateTraining(g, registry);
    const dataErr = errors.find((e) => e.message.match(/no data node/i));
    expect(dataErr).toBeDefined();
  });

  it('errors when no loss node is present', () => {
    // Graph with data + linear + optimizer but no loss
    // data -> linear is fine for forward (linear.in connected, data has no required inputs)
    // But optimizer.loss port is required and disconnected -> forward fails
    // We need opt to not have a required unconnected port. Build a minimal graph:
    // data -> linear (just these two, both forward-valid) then check training
    const registry = buildRegistry();

    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));

    const errors = validateTraining(g, registry);
    const lossErr = errors.find((e) => e.message.match(/no loss node/i));
    expect(lossErr).toBeDefined();
  });

  it('errors when no optimizer node is present', () => {
    // data -> linear -> loss (all ports connected, forward passes, but no optimizer)
    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));
    addEdge(g, createEdge('e2', 'linear', 'out', 'loss', 'predictions'));
    addEdge(g, createEdge('e3', 'data', 'labels', 'loss', 'labels'));

    const errors = validateTraining(g, buildRegistry());
    const optErr = errors.find((e) => e.message.match(/no optimizer node/i));
    expect(optErr).toBeDefined();
  });

  it('errors when multiple data nodes are present', () => {
    const g = buildValidTrainingGraph();
    // Add a second data node
    addNode(g, createNode('data2', 'data.cifar10'));

    const errors = validateTraining(g, buildRegistry());
    const multiErr = errors.find((e) => e.message.match(/multiple data nodes/i));
    expect(multiErr).toBeDefined();
  });

  it('errors when multiple optimizer nodes are present', () => {
    const g = buildValidTrainingGraph();
    // Add a second optimizer connected to loss
    addNode(g, createNode('opt2', 'ml.optimizers.adam'));
    addEdge(g, createEdge('e-extra', 'loss', 'loss', 'opt2', 'loss'));

    const errors = validateTraining(g, buildRegistry());
    const multiErr = errors.find((e) => e.message.match(/multiple optimizer/i));
    expect(multiErr).toBeDefined();
  });

  it('returns no errors for a valid training graph', () => {
    const g = buildValidTrainingGraph();
    const errors = validateTraining(g, buildRegistry());
    expect(errors).toHaveLength(0);
  });

  it('errors when loss predictions port is not connected', () => {
    // Valid graph but omit the predictions edge on loss
    // loss.predictions is required -> forward will catch it
    // That's the correct behavior — we just check it's flagged
    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addNode(g, createNode('opt', 'ml.optimizers.sgd'));
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));
    // Skip: linear -> loss predictions
    addEdge(g, createEdge('e3', 'data', 'labels', 'loss', 'labels'));
    addEdge(g, createEdge('e4', 'loss', 'loss', 'opt', 'loss'));

    const errors = validateTraining(g, buildRegistry());
    const predErr = errors.find((e) => e.message.match(/predictions.*not connected/i));
    expect(predErr).toBeDefined();
  });

  it('errors when loss labels port is not connected', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addNode(g, createNode('opt', 'ml.optimizers.sgd'));
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));
    addEdge(g, createEdge('e2', 'linear', 'out', 'loss', 'predictions'));
    // Skip: data -> loss labels
    addEdge(g, createEdge('e4', 'loss', 'loss', 'opt', 'loss'));

    const errors = validateTraining(g, buildRegistry());
    const labelsErr = errors.find((e) => e.message.match(/labels.*not connected/i));
    expect(labelsErr).toBeDefined();
  });

  it('errors when optimizer loss port is not connected', () => {
    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addNode(g, createNode('opt', 'ml.optimizers.sgd'));
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));
    addEdge(g, createEdge('e2', 'linear', 'out', 'loss', 'predictions'));
    addEdge(g, createEdge('e3', 'data', 'labels', 'loss', 'labels'));
    // Skip: loss -> opt

    const errors = validateTraining(g, buildRegistry());
    const lossErr = errors.find((e) => e.message.match(/loss.*not connected/i));
    expect(lossErr).toBeDefined();
  });

  it('errors when loss is not reachable from data node', () => {
    // Build a graph where data and loss both exist and all required ports
    // are connected, but loss is NOT downstream of data.
    // Use a custom source node to feed data into loss so forward validation passes
    // but the reachability check from the data node still fails.
    const registry = buildRegistry();
    registry.register(makeDef('custom.source', 'Source', [
      outputPort('out', 'Output'),
      outputPort('labels', 'Labels'),
    ]));

    const g = createGraph('g', 'test');
    addNode(g, createNode('data', 'data.mnist'));
    addNode(g, createNode('linear', 'ml.layers.linear'));
    addNode(g, createNode('src', 'custom.source')); // feeds loss instead of data
    addNode(g, createNode('loss', 'ml.loss.cross_entropy'));
    addNode(g, createNode('opt', 'ml.optimizers.sgd'));

    // data -> linear (data's path, but doesn't reach loss)
    addEdge(g, createEdge('e1', 'data', 'images', 'linear', 'in'));
    // src -> loss (loss gets its inputs from src, not data)
    addEdge(g, createEdge('e2', 'src', 'out', 'loss', 'predictions'));
    addEdge(g, createEdge('e3', 'src', 'labels', 'loss', 'labels'));
    addEdge(g, createEdge('e4', 'loss', 'loss', 'opt', 'loss'));

    const errors = validateTraining(g, registry);
    const reachErr = errors.find((e) => e.message.match(/not reachable/i));
    expect(reachErr).toBeDefined();
  });
});
