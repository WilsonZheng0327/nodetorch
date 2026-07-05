// Auto-layout — pure graph math (Layer 1, no React/DOM, no ML knowledge).
// Assigns each node a column by its longest path from a root, then spaces
// columns/rows evenly. Returns positions; mutates nothing.

import type { Graph } from './graph';

export interface LayoutOptions {
  /** Horizontal gap between columns (pixels). */
  colGap?: number;
  /** Vertical gap between nodes in a column (pixels). */
  rowGap?: number;
}

/**
 * Compute an auto-layout for a graph: longest-path layering left-to-right so
 * connected nodes land in adjacent columns, disconnected nodes fall in column 0,
 * and nodes within a column keep their relative top-to-bottom order.
 *
 * @returns nodeId → { x, y } for every node in the graph.
 */
export function computeLayout(
  graph: Graph,
  opts: LayoutOptions = {},
): Map<string, { x: number; y: number }> {
  const { colGap = 250, rowGap = 120 } = opts;
  const positions = new Map<string, { x: number; y: number }>();
  if (graph.nodes.size === 0) return positions;

  // Adjacency + in-degree.
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const node of graph.nodes.values()) {
    inDegree.set(node.id, 0);
    adj.set(node.id, []);
  }
  for (const edge of graph.edges) {
    inDegree.set(edge.target.nodeId, (inDegree.get(edge.target.nodeId) ?? 0) + 1);
    adj.get(edge.source.nodeId)?.push(edge.target.nodeId);
  }

  // Longest-path layering (Kahn's algorithm, tracking max depth per node).
  const depth = new Map<string, number>();
  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) {
      depth.set(id, 0);
      queue.push(id);
    }
  }
  // A fully-cyclic or empty-root graph: seed every node at column 0.
  if (queue.length === 0) {
    for (const id of graph.nodes.keys()) {
      depth.set(id, 0);
      queue.push(id);
    }
  }

  while (queue.length > 0) {
    const id = queue.shift()!;
    const d = depth.get(id) ?? 0;
    for (const tgt of adj.get(id) ?? []) {
      const newD = d + 1;
      if (newD > (depth.get(tgt) ?? 0)) depth.set(tgt, newD);
      const remaining = (inDegree.get(tgt) ?? 1) - 1;
      inDegree.set(tgt, remaining);
      if (remaining === 0) queue.push(tgt);
    }
  }

  // Disconnected / unvisited nodes go to column 0.
  for (const id of graph.nodes.keys()) {
    if (!depth.has(id)) depth.set(id, 0);
  }

  // Group by column.
  const columns = new Map<number, string[]>();
  for (const [id, d] of depth) {
    if (!columns.has(d)) columns.set(d, []);
    columns.get(d)!.push(id);
  }

  // Space out, preserving each column's current top-to-bottom order.
  for (const [col, ids] of columns) {
    ids.sort(
      (a, b) => (graph.nodes.get(a)?.position.y ?? 0) - (graph.nodes.get(b)?.position.y ?? 0),
    );
    const startY = -((ids.length - 1) * rowGap) / 2;
    ids.forEach((id, i) => {
      positions.set(id, { x: col * colGap, y: startY + i * rowGap });
    });
  }

  return positions;
}
