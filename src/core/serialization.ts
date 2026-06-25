// Graph serialization — the on-disk / over-the-wire format for a Graph.
//
// `SerializedGraph` (version '1.0') is the format used for: saving/loading graph
// files, the undo/redo snapshots, the clipboard, the saved-block library, and the
// payload sent to the backend. It is a plain-JSON projection of a Graph (Layer 1)
// with nested `subgraph` support for composite nodes — no React/UI state.
//
// `validateSerializedGraph` is a pure pre-flight check: run it before
// `deserializeGraph` so a malformed, outdated, or unknown-node file fails with a
// readable message instead of throwing half-way through `createNode`/`addEdge`.

import {
  type Graph,
  createGraph,
  createNode,
  addNode,
  addEdge,
} from './graph';

/** A node as stored in a SerializedGraph (positions + properties, no runtime state). */
export interface SerializedNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  properties: Record<string, any>;
  subgraph?: SerializedGraphData;
}

/** The inner graph payload: identity, nodes, and edges. */
export interface SerializedGraphData {
  id: string;
  name: string;
  nodes: SerializedNode[];
  edges: { id: string; source: { nodeId: string; portId: string }; target: { nodeId: string; portId: string } }[];
}

/** The versioned top-level wrapper actually written to files / sent to the backend. */
export interface SerializedGraph {
  version: '1.0';
  graph: SerializedGraphData;
}

/** The format version this build reads and writes. */
export const SERIALIZED_GRAPH_VERSION = '1.0';

/** Project a Graph into the plain-JSON `SerializedGraphData` (recurses into subgraphs). */
export function serializeGraphData(graph: Graph): SerializedGraphData {
  return {
    id: graph.id,
    name: graph.name,
    nodes: Array.from(graph.nodes.values()).map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      properties: n.properties,
      ...(n.subgraph ? { subgraph: serializeGraphData(n.subgraph) } : {}),
    })),
    edges: graph.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
    })),
  };
}

/** Wrap a Graph in the versioned top-level envelope. */
export function serializeGraph(graph: Graph): SerializedGraph {
  return { version: SERIALIZED_GRAPH_VERSION, graph: serializeGraphData(graph) };
}

/** Rebuild a Graph from `SerializedGraphData` (recurses into subgraphs).
 *  Assumes the data is structurally sound — call `validateSerializedGraph` first
 *  for untrusted input, since `createNode`/`addEdge` throw on bad references. */
export function deserializeGraphData(data: SerializedGraphData): Graph {
  const graph = createGraph(data.id, data.name);
  for (const n of data.nodes) {
    const node = createNode(n.id, n.type, n.position, n.properties);
    if (n.subgraph) {
      node.subgraph = deserializeGraphData(n.subgraph);
    }
    addNode(graph, node);
  }
  for (const e of data.edges) {
    addEdge(graph, { id: e.id, source: e.source, target: e.target });
  }
  return graph;
}

/** Rebuild a Graph from the versioned envelope. */
export function deserializeGraph(data: SerializedGraph): Graph {
  return deserializeGraphData(data.graph);
}

/**
 * Structurally validate a parsed graph payload BEFORE deserializing it. Returns a
 * list of human-readable problems (empty array = safe to deserialize). Catches the
 * cases that would otherwise throw mid-build or silently corrupt the canvas: wrong
 * version, missing/duplicate node ids, edges pointing at non-existent nodes, and
 * (when `isKnownType` is supplied) node types this build doesn't recognize.
 *
 * `isKnownType` is injected rather than importing the node registry so this stays
 * a pure Layer-1 function with no dependency on the ML/domain layers.
 */
export function validateSerializedGraph(
  data: unknown,
  isKnownType?: (type: string) => boolean,
): string[] {
  if (!data || typeof data !== 'object') {
    return ['Not a valid graph file — expected a JSON object.'];
  }
  const top = data as Partial<SerializedGraph>;
  const issues: string[] = [];

  if (top.version !== SERIALIZED_GRAPH_VERSION) {
    issues.push(
      `Unsupported graph version ${JSON.stringify(top.version)} — this build reads "${SERIALIZED_GRAPH_VERSION}".`,
    );
  }

  const g = top.graph as SerializedGraphData | undefined;
  if (!g || typeof g !== 'object' || !Array.isArray(g.nodes)) {
    issues.push('Graph data is missing its "nodes" list.');
    return issues; // nothing more can be checked
  }

  validateGraphData(g, isKnownType, issues, 'graph');
  return issues;
}

/** Recursive worker for `validateSerializedGraph` — checks one graph level and its subgraphs. */
function validateGraphData(
  g: SerializedGraphData,
  isKnownType: ((type: string) => boolean) | undefined,
  issues: string[],
  path: string,
): void {
  const ids = new Set<string>();
  for (const n of g.nodes ?? []) {
    if (!n || typeof n.id !== 'string' || typeof n.type !== 'string') {
      issues.push(`${path}: a node is missing its id or type.`);
      continue;
    }
    if (ids.has(n.id)) issues.push(`${path}: duplicate node id "${n.id}".`);
    ids.add(n.id);
    if (isKnownType && !isKnownType(n.type)) {
      issues.push(`${path}: unknown node type "${n.type}" (node "${n.id}") — it may come from a newer version.`);
    }
    if (n.subgraph) validateGraphData(n.subgraph, isKnownType, issues, `${path} › ${n.id}`);
  }

  const seen = new Set<string>();
  for (const e of g.edges ?? []) {
    const s = e?.source, t = e?.target;
    if (!s || !t) {
      issues.push(`${path}: an edge is missing its source or target.`);
      continue;
    }
    if (!ids.has(s.nodeId)) issues.push(`${path}: edge references missing node "${s.nodeId}".`);
    if (!ids.has(t.nodeId)) issues.push(`${path}: edge references missing node "${t.nodeId}".`);
    const key = `${s.nodeId}.${s.portId}->${t.nodeId}.${t.portId}`;
    if (seen.has(key)) issues.push(`${path}: duplicate edge ${key}.`);
    seen.add(key);
  }
}
