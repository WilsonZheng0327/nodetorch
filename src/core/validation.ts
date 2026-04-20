// Graph validation — pre-flight checks before running forward pass or training.
// Returns a list of errors. Empty list = graph is valid.
//
// Two levels:
//   validateForward() — checks every non-optional input port has a connection
//   validateTraining() — all forward checks + requires data/loss/optimizer nodes,
//                        validates loss has predictions+labels, optimizer has loss,
//                        checks reachability from data to loss
//
// Called by useGraph before sending to backend — gives instant feedback.

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

const LOSS_TYPES = ['ml.loss.cross_entropy', 'ml.loss.mse', 'ml.loss.vae', 'ml.loss.gan'];
const OPTIMIZER_TYPES = ['ml.optimizers.sgd', 'ml.optimizers.adam', 'ml.optimizers.adamw'];
const DATA_TYPES = ['data.mnist', 'data.cifar10', 'data.cifar100', 'data.fashion_mnist', 'data.imdb', 'data.ag_news'];
const GAN_INPUT_TYPES = ['ml.gan.noise_input'];
const DIFFUSION_TYPES = ['ml.diffusion.noise_scheduler', 'ml.diffusion.timestep_embed'];

export function validateTraining(graph: Graph, registry: NodeRegistry): ValidationError[] {
  // Start with forward validation
  const errors = validateForward(graph, registry);
  if (errors.length > 0) return errors;

  // Detect GAN mode
  const ganLossNodes = [...graph.nodes.values()].filter((n) => n.type === 'ml.loss.gan');
  const isGanMode = ganLossNodes.length > 0;

  // Detect diffusion mode
  const diffusionSchedulerNodes = [...graph.nodes.values()].filter((n) => n.type === 'ml.diffusion.noise_scheduler');
  const isDiffusionMode = diffusionSchedulerNodes.length > 0;

  // Must have exactly one data node
  const dataNodes = [...graph.nodes.values()].filter((n) => DATA_TYPES.includes(n.type));
  if (dataNodes.length === 0) {
    errors.push({ message: 'No data node — add a dataset (e.g. MNIST)' });
  }
  if (dataNodes.length > 1) {
    errors.push({ message: 'Multiple data nodes — only one dataset is supported per graph' });
  }

  // Must have at least one loss node
  const lossNodes = [...graph.nodes.values()].filter((n) => LOSS_TYPES.includes(n.type));
  if (lossNodes.length === 0) {
    errors.push({ message: 'No loss node — add a loss function (e.g. CrossEntropyLoss)' });
  }

  // Must have optimizer node(s) — GAN requires exactly 2
  const optimizerNodes = [...graph.nodes.values()].filter((n) => OPTIMIZER_TYPES.includes(n.type));
  if (optimizerNodes.length === 0) {
    errors.push({ message: 'No optimizer node — add an optimizer (e.g. SGD)' });
  }
  if (isGanMode) {
    if (optimizerNodes.length !== 2) {
      errors.push({ message: 'GAN requires exactly 2 optimizer nodes — one for Generator, one for Discriminator' });
    }
    // GAN requires a noise input node
    const noiseInputNodes = [...graph.nodes.values()].filter((n) => GAN_INPUT_TYPES.includes(n.type));
    if (noiseInputNodes.length === 0) {
      errors.push({ message: 'GAN requires a Noise Input node for the generator' });
    }
    if (noiseInputNodes.length > 1) {
      errors.push({ message: 'Multiple Noise Input nodes — only one is supported per GAN' });
    }
  } else if (optimizerNodes.length > 1) {
    errors.push({ message: 'Multiple optimizer nodes — only one is supported' });
  }

  // Loss node must have required inputs connected
  for (const lossNode of lossNodes) {
    const lossName = registry.get(lossNode.type)?.displayName ?? lossNode.type;

    if (lossNode.type === 'ml.loss.vae') {
      // VAE loss has 4 required inputs: reconstruction, original, mean, logvar
      const requiredPorts = ['reconstruction', 'original', 'mean', 'logvar'];
      for (const portId of requiredPorts) {
        const connected = graph.edges.some(
          (e) => e.target.nodeId === lossNode.id && e.target.portId === portId,
        );
        if (!connected) {
          errors.push({ nodeId: lossNode.id, message: `${lossName}: ${portId} port not connected` });
        }
      }
    } else if (lossNode.type === 'ml.loss.gan') {
      // GAN loss has 2 required inputs: real_scores and fake_scores
      const requiredPorts = ['real_scores', 'fake_scores'];
      for (const portId of requiredPorts) {
        const connected = graph.edges.some(
          (e) => e.target.nodeId === lossNode.id && e.target.portId === portId,
        );
        if (!connected) {
          errors.push({ nodeId: lossNode.id, message: `${lossName}: ${portId} port not connected` });
        }
      }
    } else {
      // Standard losses: predictions + labels
      const predConnected = graph.edges.some(
        (e) => e.target.nodeId === lossNode.id && e.target.portId === 'predictions',
      );
      const labelsConnected = graph.edges.some(
        (e) => e.target.nodeId === lossNode.id && e.target.portId === 'labels',
      );

      if (!predConnected) {
        errors.push({ nodeId: lossNode.id, message: `${lossName}: predictions port not connected` });
      }
      if (!labelsConnected) {
        errors.push({ nodeId: lossNode.id, message: `${lossName}: labels port not connected` });
      }
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

  // Optimizer sanity: epochs and learning rate must be positive
  for (const optNode of optimizerNodes) {
    const epochs = optNode.properties.epochs ?? optNode.properties.epoch;
    const lr = optNode.properties.lr ?? optNode.properties.learningRate;
    const optName = registry.get(optNode.type)?.displayName ?? optNode.type;
    if (epochs != null && epochs <= 0) {
      errors.push({ nodeId: optNode.id, message: `${optName}: epochs must be > 0` });
    }
    if (lr != null && lr <= 0) {
      errors.push({ nodeId: optNode.id, message: `${optName}: learning rate must be > 0` });
    }
  }

  // Embedding numEmbeddings must be >= connected dataset's vocabSize
  for (const [nodeId, node] of graph.nodes) {
    if (node.type === 'ml.layers.embedding') {
      const numEmb = node.properties.numEmbeddings;
      // Find the data node feeding into this embedding (walk upstream)
      for (const dataNode of dataNodes) {
        const vocabSize = dataNode.properties.vocabSize;
        if (numEmb != null && vocabSize != null && numEmb < vocabSize) {
          errors.push({
            nodeId,
            message: `Embedding: numEmbeddings (${numEmb}) < dataset vocabSize (${vocabSize}) — will crash on out-of-range tokens`,
          });
        }
      }
    }
  }

  // Subgraph blocks should have inner connections (not empty pass-through)
  for (const [nodeId, node] of graph.nodes) {
    if (node.type === 'subgraph.block' && node.subgraph) {
      const innerEdges = node.subgraph.edges.length;
      const innerNodes = node.subgraph.nodes.size;
      // A subgraph with only input+output sentinels and no edges is empty
      if (innerNodes <= 2 || innerEdges === 0) {
        const blockName = node.properties.blockName ?? nodeId;
        errors.push({
          nodeId,
          message: `Block "${blockName}" is empty — add layers inside it or remove it`,
        });
      }
    }
  }

  // Check there's a path from data → loss (model layers exist in between)
  // For GAN mode, also check reachability from noise input
  if (dataNodes.length > 0 && lossNodes.length > 0) {
    const sourceNodes = [...dataNodes];
    // In GAN mode, also include noise input as a source
    if (isGanMode) {
      const noiseNodes = [...graph.nodes.values()].filter((n) => GAN_INPUT_TYPES.includes(n.type));
      sourceNodes.push(...noiseNodes);
    }

    const reachable = new Set<string>();
    const queue = sourceNodes.map((n) => n.id);
    for (const id of queue) reachable.add(id);

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
