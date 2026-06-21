// Executes an agent tool call against the live graph via useGraph actions —
// the same functions the inspector/palette use, so edits are validated, render
// live, and are undoable. Returns a human-readable observation string the model
// reads back (errors included, so it can self-correct).

import type * as RF from '@xyflow/react';
import type { Graph, NodeInstance } from '../../core/graph';
import type { NodeRegistry } from '../../core/nodedef';
import { validateForward, validateTraining } from '../../core/validation';

/** The subset of useGraph the tool executor needs. */
export interface GraphToolApi {
  /** Live current graph (root or active subgraph) — read fresh each call. */
  getCurrentGraph: () => Graph;
  updateProperty: (nodeId: string, key: string, value: unknown) => Promise<void>;
  addNode: (type: string, position: { x: number; y: number }, requestedId?: string) => Promise<string | undefined>;
  connect: (connection: RF.Connection) => Promise<void>;
  removeNode: (nodeId: string) => Promise<void>;
  removeEdge: (sourceId: string, sourcePort: string, targetId: string, targetPort: string) => Promise<boolean>;
  clearGraph: () => void;
  organizeGraph: () => void;
  addBlockFromTemplate: (filename: string, position: { x: number; y: number }) => Promise<void>;
  /** Descend into a subgraph.block node so further edits target its inner graph. */
  enterSubgraph: (nodeId: string) => void;
  /** Pop up one level out of the current subgraph. */
  exitSubgraph: () => void;
  /** Save a subgraph.block node to the reusable-block library. */
  saveBlock: (nodeId: string) => Promise<void>;
  isValidConnection: (connection: RF.Connection) => boolean;
}

type Args = Record<string, unknown>;
type Domain = { nodeRegistry: NodeRegistry };

/** Place a new node to the right of the rightmost existing node. */
function nextPosition(g: Graph): { x: number; y: number } {
  let maxX = -Infinity;
  for (const n of g.nodes.values()) maxX = Math.max(maxX, n.position.x);
  return { x: maxX === -Infinity ? 120 : maxX + 220, y: 140 };
}

function summarizeGraph(g: Graph): string {
  const nodes = [...g.nodes.values()];
  if (!nodes.length) return '(the canvas is empty)';
  const lines: string[] = [`Nodes (${nodes.length}):`];
  for (const n of nodes) {
    const props = Object.entries(n.properties)
      .filter(([, v]) => v !== undefined && v !== '')
      .map(([k, v]) => `${k}=${v}`)
      .join(', ');
    lines.push(`- ${n.id} (${n.type})${props ? '  ' + props : ''}`);
  }
  if (g.edges.length) {
    lines.push(`Edges (${g.edges.length}):`);
    for (const e of g.edges) lines.push(`- ${e.source.nodeId}.${e.source.portId} -> ${e.target.nodeId}.${e.target.portId}`);
  }
  return lines.join('\n');
}

function describeNode(n: NodeInstance): string {
  const md = n.lastResult?.metadata as Record<string, unknown> | undefined;
  const props = Object.entries(n.properties).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(', ');
  const parts = [`${n.id} (${n.type})`, `props: ${props || '(none)'}`];
  if (md?.outputShape) parts.push(`outputShape: [${(md.outputShape as unknown[]).join(', ')}]`);
  if (md?.paramCount != null) parts.push(`params: ${md.paramCount}`);
  if (md?.error) parts.push(`ERROR: ${md.error}`);
  return parts.join('\n');
}

export async function executeGraphTool(graph: GraphToolApi, domain: Domain, name: string, args: Args): Promise<string> {
  try {
    switch (name) {
      case 'set_node_property':
        return await setNodeProperty(graph, domain, args);
      case 'add_node':
        return await addNode(graph, domain, args);
      case 'connect':
        return await connect(graph, args);
      case 'remove_node':
        return await removeNode(graph, args);
      case 'remove_edge':
        return await removeEdge(graph, args);
      case 'clear_graph':
        graph.clearGraph();
        return 'ok: cleared the canvas (all nodes and edges removed)';
      case 'organize_layout':
        graph.organizeGraph();
        return 'ok: re-arranged the node layout';
      case 'add_block':
        return await addBlock(graph, args);
      case 'enter_block':
        return enterBlock(graph, args);
      case 'exit_block':
        graph.exitSubgraph();
        return 'ok: exited to the parent canvas';
      case 'save_block':
        return await saveBlock(graph, args);
      case 'get_graph':
        return summarizeGraph(graph.getCurrentGraph());
      case 'get_node':
        return getNode(graph, args);
      case 'validate':
        return validate(graph, domain, args);
      default:
        return `error: unknown tool "${name}"`;
    }
  } catch (e) {
    return `error: ${e instanceof Error ? e.message : String(e)}`;
  }
}

async function setNodeProperty(graph: GraphToolApi, domain: Domain, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const key = String(args.key ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  const def = domain.nodeRegistry.get(node.type);
  const props = def?.getProperties() ?? [];
  if (def && !props.some((p) => p.id === key)) {
    return `error: "${node.type}" has no property "${key}". Valid keys: ${props.map((p) => p.id).join(', ') || '(none)'}`;
  }
  await graph.updateProperty(nodeId, key, args.value);
  return `ok: set ${nodeId}.${key} = ${JSON.stringify(args.value)}`;
}

async function addNode(graph: GraphToolApi, domain: Domain, args: Args) {
  const type = String(args.type ?? '');
  if (!domain.nodeRegistry.get(type)) return `error: unknown node type "${type}"`;
  const requestedId = typeof args.id === 'string' && args.id ? args.id : undefined;
  const id = await graph.addNode(type, nextPosition(graph.getCurrentGraph()), requestedId);
  if (!id) return `error: failed to add node "${type}"`;
  if (args.properties && typeof args.properties === 'object') {
    for (const [k, v] of Object.entries(args.properties as Args)) {
      await graph.updateProperty(id, k, v);
    }
  }
  return `ok: added ${type} as "${id}"`;
}

async function connect(graph: GraphToolApi, args: Args) {
  const sourceId = String(args.sourceId ?? '');
  const targetId = String(args.targetId ?? '');
  const sourcePort = String(args.sourcePort ?? '');
  const targetPort = String(args.targetPort ?? '');
  const g = graph.getCurrentGraph();
  if (!g.nodes.has(sourceId)) return `error: no node "${sourceId}"`;
  if (!g.nodes.has(targetId)) return `error: no node "${targetId}"`;
  const connection: RF.Connection = { source: sourceId, sourceHandle: sourcePort, target: targetId, targetHandle: targetPort };
  if (!graph.isValidConnection(connection)) {
    return `error: invalid connection ${sourceId}.${sourcePort} -> ${targetId}.${targetPort} (check port ids and type compatibility)`;
  }
  await graph.connect(connection);
  return `ok: connected ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`;
}

async function removeNode(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  if (!graph.getCurrentGraph().nodes.has(nodeId)) return `error: no node "${nodeId}"`;
  await graph.removeNode(nodeId);
  return `ok: removed ${nodeId}`;
}

async function removeEdge(graph: GraphToolApi, args: Args) {
  const sourceId = String(args.sourceId ?? '');
  const sourcePort = String(args.sourcePort ?? '');
  const targetId = String(args.targetId ?? '');
  const targetPort = String(args.targetPort ?? '');
  const ok = await graph.removeEdge(sourceId, sourcePort, targetId, targetPort);
  return ok
    ? `ok: disconnected ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`
    : `error: no edge ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`;
}

async function addBlock(graph: GraphToolApi, args: Args) {
  const filename = String(args.filename ?? '');
  if (!filename) return 'error: filename required';
  await graph.addBlockFromTemplate(filename, nextPosition(graph.getCurrentGraph()));
  return `ok: added block "${filename}"`;
}

function enterBlock(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  if (node.type !== 'subgraph.block' || !node.subgraph) {
    return `error: "${nodeId}" is not a custom block — only subgraph.block nodes can be entered`;
  }
  graph.enterSubgraph(nodeId);
  return `ok: entered block "${nodeId}" — add_node/connect now edit its inner graph (use exit_block when done)`;
}

async function saveBlock(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  if (node.type !== 'subgraph.block' || !node.subgraph) {
    return `error: "${nodeId}" is not a custom block — only subgraph.block nodes can be saved`;
  }
  const name = typeof args.name === 'string' && args.name ? args.name : undefined;
  if (name) await graph.updateProperty(nodeId, 'blockName', name);
  await graph.saveBlock(nodeId);
  return `ok: saved "${nodeId}" to your block library${name ? ` as "${name}"` : ''}`;
}

function getNode(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  return describeNode(node);
}

function validate(graph: GraphToolApi, domain: Domain, args: Args) {
  const mode = args.mode === 'training' ? 'training' : 'forward';
  const g = graph.getCurrentGraph();
  const errors = mode === 'training' ? validateTraining(g, domain.nodeRegistry) : validateForward(g, domain.nodeRegistry);
  if (!errors.length) return `ok: no ${mode} validation issues`;
  return `${errors.length} ${mode} issue(s):\n` + errors.map((e) => `- ${e.nodeId ? e.nodeId + ': ' : ''}${e.message}`).join('\n');
}
