// Layer 3: Execution Engine
// Runs the graph in different modes. Doesn't know what the modes do —
// just how to propagate execution through nodes in the right order.

import { type Graph, type ExecutionResult, topologicalSort } from './graph';

// --- Execution Mode ---

/**
 * Defines how a particular execution mode behaves.
 * Layer 5 registers modes (shape, forward, train) via engine.registerMode().
 */
export interface ExecutionModeDefinition {
  /** Mode identifier, e.g. "shape", "forward", "train" */
  id: string;
  /** Display name, e.g. "Shape Inference" */
  label: string;
  /**
   * "eager" = auto-run whenever nodes are dirty (used by shape mode).
   * "manual" = only run when explicitly triggered (used by forward/train).
   */
  propagation: 'eager' | 'manual';
  /** If true, skip nodes that aren't dirty. False for train (every step is a fresh pass). */
  caching: boolean;
  /**
   * Which key to look up in a node's executors map.
   * Usually same as id ("shape"→"shape"), but could differ — e.g. a "debug" mode
   * could reuse the "forward" executor with different caching/propagation settings.
   */
  executorKey: string;
}

/**
 * The function that actually runs on a node during execution.
 * Layer 3 defines this interface. Layer 4 stores executors on NodeDefinitions.
 * Layer 5 implements them (e.g. shape math for Conv2d).
 */
export interface Executor {
  execute(context: ExecutionContext): Promise<ExecutionResult>;
}

/**
 * Everything an executor needs to do its job.
 * Built by the engine from the graph state and passed to each executor.
 */
export interface ExecutionContext {
  /** portId → data from upstream node's outputs. e.g. { in: [1, 64, 26, 26] } */
  inputs: Record<string, any>;
  /** This node's current property values. e.g. { kernelSize: 3, padding: 0 } */
  properties: Record<string, any>;
  /** Persistent state across executions (weights, cached tensors) */
  state: any;
  /** Which mode is running, e.g. "shape" or "forward" */
  mode: string;
}

// --- Engine ---

/**
 * A function that finds the right executor for a node type and mode.
 * Layer 4's NodeRegistry provides this — the engine doesn't import the registry directly.
 * Returns undefined if the node doesn't support this mode (e.g. optimizer has no shape executor).
 */
export type ExecutorLookup = (nodeType: string, executorKey: string) => Executor | undefined;

/**
 * Runs execution modes on a graph. Walks nodes in topological order,
 * calls each node's executor, and stores results.
 * Doesn't know what the executors do — just when and in what order to call them.
 */
export class ExecutionEngine {
  /** modeId → mode definition */
  private modes = new Map<string, ExecutionModeDefinition>();

  /** Register a new execution mode. Called by Layer 5 at startup. */
  registerMode(mode: ExecutionModeDefinition): void {
    if (this.modes.has(mode.id)) {
      throw new Error(`Execution mode "${mode.id}" already registered`);
    }
    this.modes.set(mode.id, mode);
  }

  /** Look up a mode definition by id. */
  getMode(id: string): ExecutionModeDefinition | undefined {
    return this.modes.get(id);
  }

  /**
   * Execute a mode on the graph.
   *
   * Mode determines:
   * 1. Which executor to call on each node (mode.executorKey) — e.g., "shape" runs
   *    shape math in TS, "forward" sends data to PyTorch. Same engine loop either way.
   * 2. Whether to skip clean nodes (mode.caching) — shape/forward skip unchanged nodes,
   *    train re-runs everything (each training step needs a fresh forward pass).
   */
  async execute(graph: Graph, modeId: string, lookupExecutor: ExecutorLookup): Promise<void> {
    const mode = this.modes.get(modeId);
    if (!mode) {
      throw new Error(`Unknown execution mode "${modeId}"`);
    }

    const order = topologicalSort(graph);

    for (const nodeId of order) {
      const node = graph.nodes.get(nodeId)!;

      // If caching is on, skip clean nodes
      if (mode.caching && !node.dirty) continue;

      // Gather inputs: for each incoming edge, read the upstream node's output.
      const inputs: Record<string, any> = {};
      for (const edge of graph.edges) {
        if (edge.target.nodeId !== nodeId) continue;

        const sourceNode = graph.nodes.get(edge.source.nodeId);
        if (sourceNode?.lastResult?.outputs) {
          inputs[edge.target.portId] = sourceNode.lastResult.outputs[edge.source.portId];
        }
      }

      // --- Subgraph nodes: recurse into the inner graph ---
      if (node.subgraph) {
        // 1. Inject inputs into the GraphInput sentinel(s) inside the subgraph
        for (const [, innerNode] of node.subgraph.nodes) {
          if (innerNode.type === 'subgraph.input') {
            innerNode.lastResult = {
              outputs: inputs,
              metadata: {
                shapes: Object.entries(inputs).map(([key, val]) => ({
                  label: key,
                  value: Array.isArray(val) ? val : String(val),
                })),
              },
            };
            innerNode.dirty = false;
          } else {
            innerNode.dirty = true;
          }
        }

        // 2. Recursively execute the inner graph
        await this.execute(node.subgraph, modeId, lookupExecutor);

        // 3. Read results from the GraphOutput sentinel(s)
        const outputs: Record<string, any> = {};
        for (const [, innerNode] of node.subgraph.nodes) {
          if (innerNode.type === 'subgraph.output' && innerNode.lastResult?.outputs) {
            Object.assign(outputs, innerNode.lastResult.outputs);
          }
        }

        // Count total params from all inner nodes
        let totalParams = 0;
        const paramParts: string[] = [];
        for (const [, innerNode] of node.subgraph.nodes) {
          const pc = innerNode.lastResult?.metadata?.paramCount;
          if (pc) {
            totalParams += pc;
            const shortName = innerNode.type.split('.').pop() ?? innerNode.type;
            paramParts.push(`${shortName}: ${pc.toLocaleString()}`);
          }
        }

        node.lastResult = {
          outputs,
          metadata: {
            outputShape: Object.values(outputs)[0],
            paramCount: totalParams || undefined,
            paramBreakdown: paramParts.length > 0 ? paramParts.join(' + ') + ` = ${totalParams.toLocaleString()}` : undefined,
            shapes: Object.entries(outputs).map(([key, val]) => ({
              label: key,
              value: Array.isArray(val) ? val : String(val),
            })),
          },
        };
        node.dirty = false;
        continue;
      }

      // --- Normal nodes ---

      // Look up this node's executor for the current mode.
      // Not all nodes support all executors —
      // e.g. optimizer isn't used during shape calculation & forward pass.
      const executor = lookupExecutor(node.type, mode.executorKey);
      if (!executor) continue;

      // Run the executor
      const context: ExecutionContext = {
        inputs,
        properties: node.properties,
        state: node.state,
        mode: modeId,
      };

      const result = await executor.execute(context);

      // Store result on the node
      node.lastResult = result;
      if (result.state !== undefined) {
        node.state = result.state;
      }
      node.dirty = false;
    }
  }
}
