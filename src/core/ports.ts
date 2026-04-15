// Port resolution — resolves the effective ports for any node instance.
// For regular nodes, reads from the NodeDefinition.
// For subgraph nodes, derives ports from inner GraphInput/GraphOutput sentinels.
//
// Used by: EngineNode (rendering), isValidConnection (validation),
//          pruneInvalidEdges (cleanup). Centralizes the "what ports
//          does this node have?" question so subgraph support is transparent.

import type { NodeInstance, Graph } from './graph';
import type { PortDefinition, NodeRegistry } from './nodedef';

/**
 * Get the effective ports for a node instance.
 * Handles subgraph nodes by reading their inner graph's sentinels.
 */
export function getNodePorts(node: NodeInstance, registry: NodeRegistry): PortDefinition[] {
  if (node.subgraph) {
    return getSubgraphPorts(node.subgraph, registry);
  }

  const def = registry.get(node.type);
  if (!def) return [];
  return def.getPorts(node.properties);
}

/**
 * Derive a subgraph node's external ports from its inner graph's sentinels.
 * GraphInput output ports → SubGraph input ports
 * GraphOutput input ports → SubGraph output ports
 */
export function getSubgraphPorts(subgraph: Graph, registry: NodeRegistry): PortDefinition[] {
  const ports: PortDefinition[] = [];

  for (const [, node] of subgraph.nodes) {
    const def = registry.get(node.type);
    if (!def) continue;

    if (node.type === 'subgraph.input') {
      for (const p of def.getPorts(node.properties)) {
        ports.push({ ...p, direction: 'input', allowMultiple: false });
      }
    }

    if (node.type === 'subgraph.output') {
      for (const p of def.getPorts(node.properties)) {
        ports.push({ ...p, direction: 'output', allowMultiple: true });
      }
    }
  }

  return ports;
}
