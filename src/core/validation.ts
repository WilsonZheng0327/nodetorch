// Graph validation — pre-flight checks before running forward pass or training.
// Returns a list of errors. Empty list = graph is valid.

import type { Graph } from './graph';
import type { NodeRegistry } from './nodedef';

export interface ValidationError {
  nodeId?: string;  // which node has the problem, if applicable
  message: string;
}

// --- Forward pass checks ---

export function validateForward(graph: Graph, registry: NodeRegistry): ValidationError[] {
  const errors: ValidationError[] = [];

  if (graph.nodes.size === 0) {
    errors.push({ message: 'No nodes in graph' });
    return errors;
  }

  if (graph.edges.length === 0) {
    errors.push({ message: 'No connections — connect nodes before running' });
    return errors;
  }

  // Check every non-optional input port has a connection
  for (const [nodeId, node] of graph.nodes) {
    const def = registry.get(node.type);
    if (!def) {
      errors.push({ nodeId, message: `Unknown node type: ${node.type}` });
      continue;
    }

    const ports = def.getPorts(node.properties);
    for (const port of ports) {
      if (port.direction === 'input' && !port.optional) {
        const connected = graph.edges.some(
          (e) => e.target.nodeId === nodeId && e.target.portId === port.id,
        );
        if (!connected) {
          errors.push({
            nodeId,
            message: `${def.displayName}: "${port.name}" port is not connected`,
          });
        }
      }
    }
  }

  return errors;
}

// --- Training checks (includes all forward checks + training-specific) ---

const LOSS_TYPES = ['ml.loss.cross_entropy', 'ml.loss.mse'];
const OPTIMIZER_TYPES = ['ml.optimizers.sgd', 'ml.optimizers.adam'];
const DATA_TYPES = ['data.mnist', 'data.cifar100'];

export function validateTraining(graph: Graph, registry: NodeRegistry): ValidationError[] {
  // Start with forward validation
  const errors = validateForward(graph, registry);
  if (errors.length > 0) return errors;

  // Must have exactly one data node
  const dataNodes = [...graph.nodes.values()].filter((n) => DATA_TYPES.includes(n.type));
  if (dataNodes.length === 0) {
    errors.push({ message: 'No data node — add a dataset (e.g. MNIST)' });
  }

  // Must have at least one loss node
  const lossNodes = [...graph.nodes.values()].filter((n) => LOSS_TYPES.includes(n.type));
  if (lossNodes.length === 0) {
    errors.push({ message: 'No loss node — add a loss function (e.g. CrossEntropyLoss)' });
  }

  // Must have exactly one optimizer node
  const optimizerNodes = [...graph.nodes.values()].filter((n) => OPTIMIZER_TYPES.includes(n.type));
  if (optimizerNodes.length === 0) {
    errors.push({ message: 'No optimizer node — add an optimizer (e.g. SGD)' });
  }
  if (optimizerNodes.length > 1) {
    errors.push({ message: 'Multiple optimizer nodes — only one is supported' });
  }

  // Loss node must have both predictions and labels connected
  for (const lossNode of lossNodes) {
    const predConnected = graph.edges.some(
      (e) => e.target.nodeId === lossNode.id && e.target.portId === 'predictions',
    );
    const labelsConnected = graph.edges.some(
      (e) => e.target.nodeId === lossNode.id && e.target.portId === 'labels',
    );
    const lossName = registry.get(lossNode.type)?.displayName ?? lossNode.type;

    if (!predConnected) {
      errors.push({ nodeId: lossNode.id, message: `${lossName}: predictions port not connected` });
    }
    if (!labelsConnected) {
      errors.push({ nodeId: lossNode.id, message: `${lossName}: labels port not connected` });
    }
  }

  // Optimizer must have loss connected
  for (const optNode of optimizerNodes) {
    const lossConnected = graph.edges.some(
      (e) => e.target.nodeId === optNode.id && e.target.portId === 'loss',
    );
    const optName = registry.get(optNode.type)?.displayName ?? optNode.type;

    if (!lossConnected) {
      errors.push({ nodeId: optNode.id, message: `${optName}: loss port not connected` });
    }
  }

  // Check there's a path from data → loss (model layers exist in between)
  if (dataNodes.length > 0 && lossNodes.length > 0) {
    const reachable = new Set<string>();
    const queue = dataNodes.map((n) => n.id);
    reachable.add(queue[0]);

    while (queue.length > 0) {
      const current = queue.shift()!;
      for (const edge of graph.edges) {
        if (edge.source.nodeId === current && !reachable.has(edge.target.nodeId)) {
          reachable.add(edge.target.nodeId);
          queue.push(edge.target.nodeId);
        }
      }
    }

    for (const lossNode of lossNodes) {
      if (!reachable.has(lossNode.id)) {
        const lossName = registry.get(lossNode.type)?.displayName ?? lossNode.type;
        errors.push({ message: `${lossName} is not reachable from the data node — check connections` });
      }
    }
  }

  return errors;
}
