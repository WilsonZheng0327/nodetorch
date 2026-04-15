// Layer 1: Graph Core
// Knows about nodes, ports, edges, and graph topology.
// Does NOT know about ML, data types, or execution.

// --- Data Structures ---

/**
 * A specific node on the canvas. This is an INSTANCE — one particular Conv2d
 * with id "conv1" at position (300, 100). The DEFINITION of what Conv2d means
 * (its properties, ports, executors) lives in Layer 4's NodeDefinition.
 */
export interface NodeInstance {
  /** Unique id for this instance, e.g. "conv1" */
  id: string;
  /** References a NodeDefinition.type, e.g. "ml.layers.conv2d" */
  type: string;
  /** Canvas position */
  position: { x: number; y: number };
  /** Current property values, e.g. { kernelSize: 3, padding: 0 } */
  properties: Record<string, any>;
  /** Persistent runtime state (weights, cached tensors) */
  state: any;
  /** True = needs re-execution. Set by dirty tracking, cleared by the engine. */
  dirty: boolean;
  /** Output from the last executor run. The engine writes this, the UI reads it. */
  lastResult?: ExecutionResult;
  /** If this node is a subgraph block, this holds the inner graph. */
  subgraph?: Graph;
  /** Validation messages for this node */
  errors: ValidationMessage[];
}

/**
 * What an executor returns after running on a node.
 * Layer 1 stores this on NodeInstance.lastResult but doesn't interpret it.
 * Layer 3 (engine) populates it, Layer 6 (UI) reads it for display.
 */
export interface ExecutionResult {
  /** portId → output data. e.g. { out: [1, 64, 26, 26] } for shape mode */
  outputs: Record<string, any>;
  /** Updated persistent state, e.g. trained weights */
  state?: any;
  /** Visualization data for the UI — shapes, param counts, heatmaps, etc. */
  metadata?: Record<string, any>;
}

/** A message attached to a node for display in the UI. */
export interface ValidationMessage {
  level: 'error' | 'warning' | 'info';
  message: string;
}

/**
 * A directed connection between two ports.
 * Data flows from source → target (upstream → downstream).
 */
export interface Edge {
  id: string;
  /** Which node and port the wire starts from */
  source: { nodeId: string; portId: string };
  /** Which node and port the wire ends at */
  target: { nodeId: string; portId: string };
}

/**
 * The top-level container. Holds all nodes and edges.
 * A subgraph is also a Graph (nested inside a NodeInstance.subgraph).
 */
export interface Graph {
  id: string;
  name: string;
  /** nodeId → NodeInstance */
  nodes: Map<string, NodeInstance>;
  edges: Edge[];
  /** Arbitrary graph-level data */
  metadata: Record<string, any>;
}

// --- Graph Operations ---

/** Create an empty graph. */
export function createGraph(id: string, name: string): Graph {
  return {
    id,
    name,
    nodes: new Map(),
    edges: [],
    metadata: {},
  };
}

/** Add a node to the graph. Throws if a node with the same id already exists. */
export function addNode(graph: Graph, node: NodeInstance): void {
  if (graph.nodes.has(node.id)) {
    throw new Error(`Node "${node.id}" already exists`);
  }
  graph.nodes.set(node.id, node);
}

/** Remove a node and all edges connected to it. */
export function removeNode(graph: Graph, nodeId: string): void {
  if (!graph.nodes.has(nodeId)) {
    throw new Error(`Node "${nodeId}" not found`);
  }
  graph.edges = graph.edges.filter(
    (e) => e.source.nodeId !== nodeId && e.target.nodeId !== nodeId,
  );
  graph.nodes.delete(nodeId);
}

/** Add an edge. Validates both nodes exist and prevents duplicates. */
export function addEdge(graph: Graph, edge: Edge): void {
  if (!graph.nodes.has(edge.source.nodeId)) {
    throw new Error(`Source node "${edge.source.nodeId}" not found`);
  }
  if (!graph.nodes.has(edge.target.nodeId)) {
    throw new Error(`Target node "${edge.target.nodeId}" not found`);
  }
  const duplicate = graph.edges.find(
    (e) =>
      e.source.nodeId === edge.source.nodeId &&
      e.source.portId === edge.source.portId &&
      e.target.nodeId === edge.target.nodeId &&
      e.target.portId === edge.target.portId,
  );
  if (duplicate) {
    throw new Error(`Edge from ${edge.source.nodeId}.${edge.source.portId} to ${edge.target.nodeId}.${edge.target.portId} already exists`);
  }
  graph.edges.push(edge);
}

/** Remove an edge by id. */
export function removeEdge(graph: Graph, edgeId: string): void {
  const index = graph.edges.findIndex((e) => e.id === edgeId);
  if (index === -1) {
    throw new Error(`Edge "${edgeId}" not found`);
  }
  graph.edges.splice(index, 1);
}

/** Update a node's property and mark it + all downstream nodes dirty. */
export function setProperty(graph: Graph, nodeId: string, key: string, value: any): void {
  const node = graph.nodes.get(nodeId);
  if (!node) {
    throw new Error(`Node "${nodeId}" not found`);
  }
  node.properties[key] = value;
  markDirty(graph, nodeId);
}

// --- Topological Sort ---

/**
 * Returns node IDs in execution order (upstream before downstream).
 * Uses Kahn's algorithm. Throws if the graph has a cycle.
 */
export function topologicalSort(graph: Graph): string[] {
  // Build adjacency list: nodeId → list of downstream nodeIds
  const downstream = new Map<string, string[]>();
  // Track how many incoming edges each node has
  const inDegree = new Map<string, number>();

  for (const nodeId of graph.nodes.keys()) {
    downstream.set(nodeId, []);
    inDegree.set(nodeId, 0);
  }

  for (const edge of graph.edges) {
    downstream.get(edge.source.nodeId)!.push(edge.target.nodeId);
    inDegree.set(edge.target.nodeId, inDegree.get(edge.target.nodeId)! + 1);
  }

  // Start with nodes that have no incoming edges
  const queue: string[] = [];
  for (const [nodeId, degree] of inDegree) {
    if (degree === 0) queue.push(nodeId);
  }

  const sorted: string[] = [];
  while (queue.length > 0) {
    const nodeId = queue.shift()!;
    sorted.push(nodeId);

    for (const child of downstream.get(nodeId)!) {
      const newDegree = inDegree.get(child)! - 1;
      inDegree.set(child, newDegree);
      if (newDegree === 0) queue.push(child);
    }
  }

  if (sorted.length !== graph.nodes.size) {
    throw new Error('Graph contains a cycle');
  }

  return sorted;
}

// --- Dirty Tracking ---

/** Marks a node and all its downstream dependents as dirty (needs re-execution). */
export function markDirty(graph: Graph, nodeId: string): void {
  const node = graph.nodes.get(nodeId);
  if (!node) return;

  node.dirty = true;

  // BFS through downstream nodes
  const visited = new Set<string>([nodeId]);
  const queue = [nodeId];

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const edge of graph.edges) {
      if (edge.source.nodeId === current && !visited.has(edge.target.nodeId)) {
        visited.add(edge.target.nodeId);
        queue.push(edge.target.nodeId);
        graph.nodes.get(edge.target.nodeId)!.dirty = true;
      }
    }
  }
}

// --- Helpers ---

/** Create a new NodeInstance with default state. Starts dirty so it runs on first execution. */
export function createNode(
  id: string,
  type: string,
  position: { x: number; y: number } = { x: 0, y: 0 },
  properties: Record<string, any> = {},
): NodeInstance {
  return {
    id,
    type,
    position,
    properties,
    state: null,
    dirty: true,
    errors: [],
  };
}

/** Create an Edge between two ports. */
export function createEdge(
  id: string,
  sourceNodeId: string,
  sourcePortId: string,
  targetNodeId: string,
  targetPortId: string,
): Edge {
  return {
    id,
    source: { nodeId: sourceNodeId, portId: sourcePortId },
    target: { nodeId: targetNodeId, portId: targetPortId },
  };
}

/** Returns node IDs directly upstream of a given node (one hop back). */
export function getUpstreamNodes(graph: Graph, nodeId: string): string[] {
  return graph.edges
    .filter((e) => e.target.nodeId === nodeId)
    .map((e) => e.source.nodeId);
}

/** Returns node IDs directly downstream of a given node (one hop forward). */
export function getDownstreamNodes(graph: Graph, nodeId: string): string[] {
  return graph.edges
    .filter((e) => e.source.nodeId === nodeId)
    .map((e) => e.target.nodeId);
}
