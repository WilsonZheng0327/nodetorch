// Executes an agent tool call against the live graph via useGraph actions —
// the same functions the inspector/palette use, so edits are validated, render
// live, and are undoable. Returns a human-readable observation string the model
// reads back (errors included, so it can self-correct).

import type * as RF from '@xyflow/react';
import type { Graph } from '../../core/graph';
import type { NodeRegistry } from '../../core/nodedef';

/** The subset of useGraph the tool executor needs. */
export interface GraphToolApi {
  currentGraph: Graph;
  updateProperty: (nodeId: string, key: string, value: unknown) => Promise<void>;
  addNode: (type: string, position: { x: number; y: number }) => Promise<string | undefined>;
  connect: (connection: RF.Connection) => Promise<void>;
  removeNode: (nodeId: string) => Promise<void>;
  isValidConnection: (connection: RF.Connection) => boolean;
}

type Args = Record<string, unknown>;

/** Place a new node to the right of the rightmost existing node. */
function nextPosition(g: Graph): { x: number; y: number } {
  let maxX = -Infinity;
  for (const n of g.nodes.values()) maxX = Math.max(maxX, n.position.x);
  return { x: maxX === -Infinity ? 120 : maxX + 220, y: 140 };
}

export async function executeGraphTool(
  graph: GraphToolApi,
  domain: { nodeRegistry: NodeRegistry },
  name: string,
  args: Args,
): Promise<string> {
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
      default:
        return `error: unknown tool "${name}"`;
    }
  } catch (e) {
    return `error: ${e instanceof Error ? e.message : String(e)}`;
  }
}

async function setNodeProperty(graph: GraphToolApi, domain: { nodeRegistry: NodeRegistry }, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const key = String(args.key ?? '');
  const node = graph.currentGraph.nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  const def = domain.nodeRegistry.get(node.type);
  const props = def?.getProperties() ?? [];
  if (def && !props.some((p) => p.id === key)) {
    return `error: "${node.type}" has no property "${key}". Valid keys: ${props.map((p) => p.id).join(', ') || '(none)'}`;
  }
  await graph.updateProperty(nodeId, key, args.value);
  return `ok: set ${nodeId}.${key} = ${JSON.stringify(args.value)}`;
}

async function addNode(graph: GraphToolApi, domain: { nodeRegistry: NodeRegistry }, args: Args) {
  const type = String(args.type ?? '');
  if (!domain.nodeRegistry.get(type)) return `error: unknown node type "${type}"`;
  const id = await graph.addNode(type, nextPosition(graph.currentGraph));
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
  if (!graph.currentGraph.nodes.has(sourceId)) return `error: no node "${sourceId}"`;
  if (!graph.currentGraph.nodes.has(targetId)) return `error: no node "${targetId}"`;
  const connection: RF.Connection = {
    source: sourceId,
    sourceHandle: sourcePort,
    target: targetId,
    targetHandle: targetPort,
  };
  if (!graph.isValidConnection(connection)) {
    return `error: invalid connection ${sourceId}.${sourcePort} -> ${targetId}.${targetPort} (check port ids and type compatibility)`;
  }
  await graph.connect(connection);
  return `ok: connected ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`;
}

async function removeNode(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  if (!graph.currentGraph.nodes.has(nodeId)) return `error: no node "${nodeId}"`;
  await graph.removeNode(nodeId);
  return `ok: removed ${nodeId}`;
}
