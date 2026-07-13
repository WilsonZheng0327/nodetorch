// Bridge between our engine (Layers 1-5) and React Flow's state format.
// Owns the Graph, runs the engine, and converts to/from React Flow nodes/edges.

import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import * as RF from '@xyflow/react';
import { tutorialEvent } from './tutorial/tutorialEvent';
import {
  type Graph,
  createGraph,
  createNode,
  createEdge,
  addNode,
  removeNode,
  addEdge as addGraphEdge,
  setProperty,
  markDirty,
} from '../core/graph';
import type { DomainContext } from '../domain';
import { validateForward, validateTraining } from '../core/validation';
import { getNodePorts } from '../core/ports';
import {
  type SerializedGraph,
  serializeGraph,
  serializeGraphData,
  deserializeGraph,
  deserializeGraphData,
  validateSerializedGraph,
} from '../core/serialization';
import { apiUrl, wsUrl } from '../api/base';
import { callBackend } from '../api/client';
import type { EpochData, TestResult } from './dashboard/types';
import { friendlyError } from './friendlyError';
import { computeLayout } from '../core/layout';

// `trainingProgress` below is `EpochData[]` — the single shared per-epoch type
// (defined in ./dashboard/types, imported above), mirrored on the backend by the
// `EpochMessage` TypedDict in backend/training/base.py.

// Navigation breadcrumb entry
interface NavEntry {
  graphId: string;
  label: string;
  nodeId: string; // the subgraph node's id in the parent graph
}

// --- Conversion: our Graph → React Flow format ---

function toRFNodes(graph: Graph): RF.Node[] {
  const nodes: RF.Node[] = [];
  for (const [, node] of graph.nodes) {
    nodes.push({
      id: node.id,
      type: node.type,
      position: node.position,
      data: {
        // Spread a shallow copy so React Flow detects the change.
        // Without this, it's the same object reference (just mutated)
        // and React skips re-rendering.
        instance: { ...node },
      },
    });
  }
  return nodes;
}

function toRFEdges(graph: Graph): RF.Edge[] {
  return graph.edges.map((e) => ({
    id: e.id,
    source: e.source.nodeId,
    sourceHandle: e.source.portId,
    target: e.target.nodeId,
    targetHandle: e.target.portId,
    animated: true,
  }));
}

/** Save a blob to disk: use the File System Access API (lets the user pick a name
 *  and location) when available, otherwise fall back to a plain download. */
async function downloadBlob(
  blob: Blob,
  suggestedName: string,
  description: string,
  ext: string,
): Promise<void> {
  if ('showSaveFilePicker' in window) {
    try {
      const handle = await (window as any).showSaveFilePicker({
        suggestedName,
        types: [{ description, accept: { 'application/octet-stream': [ext] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return;
    } catch {
      return; // user cancelled the picker
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = suggestedName;
  a.click();
  URL.revokeObjectURL(url);
}

// --- The hook ---

let edgeCounter = Date.now();

export function useGraph(domain: DomainContext) {
  const graphRef = useRef<Graph>(createGraph('main', 'Main Graph'));
  const [rfNodes, setRFNodes] = useState<RF.Node[]>([]);
  const [rfEdges, setRFEdges] = useState<RF.Edge[]>([]);
  const [graphVersion, setGraphVersion] = useState(0); // bumped on every mutation to force re-renders
  const [status, setStatus] = useState<{
    type: 'idle' | 'running' | 'success' | 'error';
    message?: string;
  }>({ type: 'idle' });
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [modelStale, setModelStale] = useState(false);

  // --- Undo/Redo (refs + snapshot here, undo/redo functions after syncToRF) ---
  const undoStack = useRef<string[]>([]);
  const redoStack = useRef<string[]>([]);
  const isUndoRedo = useRef(false);
  // While a batch is active (e.g. one agent turn), individual mutations don't
  // snapshot — beginBatch takes a single snapshot so the whole batch is one undo.
  const batchActive = useRef(false);

  const snapshot = useCallback(() => {
    if (isUndoRedo.current || batchActive.current) return;
    undoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
    redoStack.current = [];
    if (undoStack.current.length > 50) undoStack.current.shift();
  }, []);

  // Group a sequence of mutations into a single undo step. beginBatch records
  // one snapshot (pre-batch state) and suppresses per-mutation snapshots until
  // endBatch. Used to make an entire agent turn revert with one Ctrl+Z.
  const beginBatch = useCallback(() => {
    if (batchActive.current) return;
    snapshot();
    batchActive.current = true;
  }, [snapshot]);

  const endBatch = useCallback(() => {
    batchActive.current = false;
  }, []);

  // --- Saved blocks ---
  interface SavedBlock {
    filename: string;
    name: string;
    description: string;
  }
  const [savedBlocks, setSavedBlocks] = useState<SavedBlock[]>([]);

  // Load saved blocks list from backend
  const refreshBlocks = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/blocks'));
      const data = await res.json();
      if (data.status === 'ok') setSavedBlocks(data.blocks);
    } catch {
      /* backend not running */
    }
  }, []);

  // Load on mount
  useEffect(() => {
    refreshBlocks();
  }, [refreshBlocks]);

  // --- Subgraph navigation ---
  // navStack tracks the path from root into nested subgraphs.
  // Empty = viewing the root graph.
  const [navStack, setNavStack] = useState<NavEntry[]>([]);

  // Get the graph currently being displayed (root or inner subgraph)
  const getCurrentGraph = useCallback((): Graph => {
    let g = graphRef.current;
    for (const entry of navStack) {
      const node = g.nodes.get(entry.nodeId);
      if (node?.subgraph) {
        g = node.subgraph;
      } else {
        break;
      }
    }
    return g;
  }, [navStack]);

  // Helper: resolve graph from a specific nav stack (not from state)
  const resolveGraph = useCallback((stack: NavEntry[]): Graph => {
    let g = graphRef.current;
    for (const entry of stack) {
      const node = g.nodes.get(entry.nodeId);
      if (node?.subgraph) g = node.subgraph;
      else break;
    }
    return g;
  }, []);

  // Immediately sync RF to a specific graph (avoids flash from async state)
  const syncToGraph = useCallback((graph: Graph) => {
    setRFNodes(toRFNodes(graph));
    setRFEdges(toRFEdges(graph));
  }, []);

  // Enter a subgraph node (double-click)
  const enterSubgraph = useCallback(
    (nodeId: string) => {
      const currentGraph = getCurrentGraph();
      const node = currentGraph.nodes.get(nodeId);
      if (!node || node.type !== 'subgraph.block' || !node.subgraph) return;

      const newStack: NavEntry[] = [
        ...navStack,
        { graphId: node.subgraph.id, label: node.properties.blockName || 'Block', nodeId },
      ];
      setNavStack(newStack);
      syncToGraph(resolveGraph(newStack));
    },
    [getCurrentGraph, navStack, resolveGraph, syncToGraph],
  );

  // Go back to a specific level in the breadcrumb
  const navigateTo = useCallback(
    (depth: number) => {
      const newStack = navStack.slice(0, depth);
      setNavStack(newStack);
      syncToGraph(resolveGraph(newStack));
    },
    [navStack, resolveGraph, syncToGraph],
  );

  // Pop up one level out of the current subgraph (no-op at the root).
  const exitSubgraph = useCallback(() => {
    if (navStack.length === 0) return;
    navigateTo(navStack.length - 1);
  }, [navStack, navigateTo]);

  // Call this whenever the graph structure or properties change
  const invalidateModel = useCallback(() => {
    if (modelTrained) {
      setModelStale(true);
    }
  }, [modelTrained]);

  // Remove edges that reference ports which don't exist on their nodes
  const pruneInvalidEdges = useCallback(() => {
    const g = getCurrentGraph();
    g.edges = g.edges.filter((e) => {
      const srcNode = g.nodes.get(e.source.nodeId);
      const tgtNode = g.nodes.get(e.target.nodeId);
      if (!srcNode || !tgtNode) return false;

      const srcPorts = getNodePorts(srcNode, domain.nodeRegistry);
      const tgtPorts = getNodePorts(tgtNode, domain.nodeRegistry);

      const srcValid = srcPorts.some((p) => p.id === e.source.portId);
      const tgtValid = tgtPorts.some((p) => p.id === e.target.portId);

      if (!srcValid || !tgtValid) {
        console.warn(
          `[nodetorch] Pruned invalid edge ${e.id}: ${e.source.portId} → ${e.target.portId}`,
        );
      }
      return srcValid && tgtValid;
    });
  }, [domain, getCurrentGraph]);

  // Sync our graph to React Flow state, preserving selection
  const syncToRF = useCallback(() => {
    pruneInvalidEdges();
    const currentGraph = getCurrentGraph();
    setRFNodes((prev) => {
      const selectedIds = new Set(prev.filter((n) => n.selected).map((n) => n.id));
      return toRFNodes(currentGraph).map((n) => ({
        ...n,
        selected: selectedIds.has(n.id),
      }));
    });
    setRFEdges(toRFEdges(currentGraph));
    setGraphVersion((v) => v + 1);
  }, [pruneInvalidEdges, getCurrentGraph]);

  // Run shape inference on the graph, then sync to React Flow
  const runShape = useCallback(async () => {
    try {
      await domain.engine.execute(graphRef.current, 'shape', (nodeType, executorKey) =>
        domain.nodeRegistry.getExecutor(nodeType, executorKey),
      );
    } catch (e) {
      console.error('[nodetorch] Shape inference failed:', e);
    }
    syncToRF();
  }, [domain, syncToRF]);

  // --- Undo/Redo actions ---
  const undo = useCallback(async () => {
    if (undoStack.current.length === 0) return;
    redoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
    const prev = undoStack.current.pop()!;
    isUndoRedo.current = true;
    graphRef.current = deserializeGraphData(JSON.parse(prev));
    // If we're inside a subgraph that no longer exists after undo, pop back to root
    let g = graphRef.current;
    let validDepth = 0;
    for (const entry of navStack) {
      const node = g.nodes.get(entry.nodeId);
      if (node?.subgraph) {
        g = node.subgraph;
        validDepth++;
      } else break;
    }
    if (validDepth < navStack.length) setNavStack(navStack.slice(0, validDepth));
    await runShape();
    isUndoRedo.current = false;
  }, [runShape, navStack]);

  const redo = useCallback(async () => {
    if (redoStack.current.length === 0) return;
    undoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
    const next = redoStack.current.pop()!;
    isUndoRedo.current = true;
    graphRef.current = deserializeGraphData(JSON.parse(next));
    let g = graphRef.current;
    let validDepth = 0;
    for (const entry of navStack) {
      const node = g.nodes.get(entry.nodeId);
      if (node?.subgraph) {
        g = node.subgraph;
        validDepth++;
      } else break;
    }
    if (validDepth < navStack.length) setNavStack(navStack.slice(0, validDepth));
    await runShape();
    isUndoRedo.current = false;
  }, [runShape, navStack]);

  // --- Connection Validation ---

  // Checks if a connection is valid before React Flow allows it.
  // Rules:
  // 1. Can't connect a node to itself
  // 2. Source must be output, target must be input (React Flow handles this)
  // 3. Data types must be compatible (Layer 2)
  // 4. Input ports with allowMultiple=false can't have more than one connection
  const isValidConnection = useCallback(
    (connection: RF.Connection | RF.Edge): boolean => {
      function reject(reason: string) {
        console.log(`[nodetorch] Connection rejected: ${reason}`, connection);
        setConnectionError(reason);
        setTimeout(() => setConnectionError(null), 3000);
        return false;
      }

      // Rule 1: no self-connections
      if (connection.source === connection.target) return reject('self-connection');

      const graph = getCurrentGraph();
      const sourceNode = graph.nodes.get(connection.source);
      const targetNode = graph.nodes.get(connection.target);
      if (!sourceNode || !targetNode) return reject('node not found');

      // Find the port definitions — use getNodePorts for subgraph support
      const sourcePorts = getNodePorts(sourceNode, domain.nodeRegistry);
      const targetPorts = getNodePorts(targetNode, domain.nodeRegistry);

      const sourcePort = sourcePorts.find((p) => p.id === connection.sourceHandle);
      const targetPort = targetPorts.find((p) => p.id === connection.targetHandle);

      const sourceName = domain.nodeRegistry.get(sourceNode.type)?.displayName ?? sourceNode.type;
      const targetName = domain.nodeRegistry.get(targetNode.type)?.displayName ?? targetNode.type;

      if (!sourcePort)
        return reject(`source port "${connection.sourceHandle}" not found on ${sourceName}`);
      if (!targetPort)
        return reject(`target port "${connection.targetHandle}" not found on ${targetName}`);

      // Rule 3: data type compatibility
      if (!domain.typeRegistry.isCompatible(sourcePort.dataType, targetPort.dataType)) {
        return reject(`type mismatch: ${sourcePort.dataType} → ${targetPort.dataType}`);
      }

      // Rule 4: check allowMultiple on target port
      if (!targetPort.allowMultiple) {
        const alreadyConnected = graph.edges.some(
          (e) =>
            e.target.nodeId === connection.target && e.target.portId === connection.targetHandle,
        );
        if (alreadyConnected) return reject(`port "${targetPort.name}" already connected`);
      }

      return true;
    },
    [domain, getCurrentGraph],
  );

  // --- Actions the UI can trigger ---

  const addNodeToGraph = useCallback(
    async (
      type: string,
      position: { x: number; y: number },
      requestedId?: string,
    ): Promise<string | undefined> => {
      const def = domain.nodeRegistry.get(type);
      if (!def) return undefined;

      // Build default properties from the definition
      const properties: Record<string, any> = {};
      for (const prop of def.getProperties()) {
        properties[prop.id] = prop.defaultValue;
      }

      // Honor a caller-requested id (e.g. an agent naming the node so it can wire
      // it in the same batch), as long as it's free; otherwise generate one.
      const id =
        requestedId && !getCurrentGraph().nodes.has(requestedId)
          ? requestedId
          : `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const node = createNode(id, type, position, properties);

      // If this is a subgraph block, create the inner graph with default sentinels
      if (type === 'subgraph.block') {
        const innerGraph = createGraph(`${id}-inner`, 'Inner Graph');
        const inputNode = createNode(
          'input',
          'subgraph.input',
          { x: 0, y: 100 },
          { portCount: 1, portNames: 'in' },
        );
        const outputNode = createNode(
          'output',
          'subgraph.output',
          { x: 400, y: 100 },
          { portCount: 1, portNames: 'out' },
        );
        addNode(innerGraph, inputNode);
        addNode(innerGraph, outputNode);
        node.subgraph = innerGraph;
      }

      snapshot();
      const currentGraph = getCurrentGraph();
      addNode(currentGraph, node);
      invalidateModel();
      await runShape();
      // Tutorial auto-detect
      if (type.startsWith('data.')) tutorialEvent('node-added-data');
      if (type === 'ml.layers.conv2d') tutorialEvent('node-added-conv2d');
      if (type === 'ml.layers.linear') tutorialEvent('node-added-linear');
      if (type === 'ml.layers.flatten') tutorialEvent('node-added-flatten');
      if (type.startsWith('ml.activations.')) tutorialEvent('node-added-activation');
      if (type.startsWith('ml.loss.')) tutorialEvent('node-added-loss');
      if (type.startsWith('ml.optimizers.')) tutorialEvent('node-added-optimizer');
      if (type.startsWith('ml.layers.batchnorm')) tutorialEvent('node-added-batchnorm');
      if (type.startsWith('ml.layers.maxpool') || type.startsWith('ml.layers.avgpool'))
        tutorialEvent('node-added-pool');
      return id;
    },
    [domain, runShape, invalidateModel, getCurrentGraph, snapshot],
  );

  const removeNodeFromGraph = useCallback(
    async (nodeId: string) => {
      snapshot();
      removeNode(getCurrentGraph(), nodeId);
      invalidateModel();
      await runShape();
    },
    [runShape, invalidateModel, getCurrentGraph, snapshot],
  );

  const connectNodes = useCallback(
    async (connection: RF.Connection) => {
      if (!connection.sourceHandle || !connection.targetHandle) {
        console.warn('[nodetorch] Connection missing handle ids', connection);
        return;
      }
      snapshot();
      const currentGraph = getCurrentGraph();
      const edgeId = `e-${edgeCounter++}`;
      const edge = createEdge(
        edgeId,
        connection.source,
        connection.sourceHandle,
        connection.target,
        connection.targetHandle,
      );
      addGraphEdge(currentGraph, edge);
      markDirty(currentGraph, connection.target);
      invalidateModel();
      await runShape();
      tutorialEvent('edge-added');
    },
    [runShape, invalidateModel, getCurrentGraph, snapshot],
  );

  const removeEdgeByEndpoints = useCallback(
    async (
      sourceId: string,
      sourcePort: string,
      targetId: string,
      targetPort: string,
    ): Promise<boolean> => {
      const g = getCurrentGraph();
      const edge = g.edges.find(
        (e) =>
          e.source.nodeId === sourceId &&
          e.source.portId === sourcePort &&
          e.target.nodeId === targetId &&
          e.target.portId === targetPort,
      );
      if (!edge) return false;
      snapshot();
      g.edges = g.edges.filter((e) => e.id !== edge.id);
      markDirty(g, targetId);
      invalidateModel();
      await runShape();
      return true;
    },
    [getCurrentGraph, snapshot, invalidateModel, runShape],
  );

  const updateProperty = useCallback(
    async (nodeId: string, key: string, value: any) => {
      snapshot();
      setProperty(getCurrentGraph(), nodeId, key, value);
      invalidateModel();
      await runShape();
      // Force a re-render even if runShape didn't change RF nodes
      // (covers edge cases where the node object is mutated in place
      // but React doesn't detect the prop change)
      setGraphVersion((v) => v + 1);
    },
    [runShape, invalidateModel, getCurrentGraph, snapshot],
  );

  // --- Serialization ---

  const saveGraph = useCallback((): string => {
    return JSON.stringify(serializeGraph(graphRef.current), null, 2);
  }, []);

  const clearGraph = useCallback(() => {
    graphRef.current = createGraph('main', 'Main Graph');
    invalidateModel();
    syncToRF();
  }, [invalidateModel, syncToRF]);

  const loadGraph = useCallback(
    async (json: string) => {
      let data: SerializedGraph;
      try {
        data = JSON.parse(json);
      } catch {
        setStatus({ type: 'error', message: 'Could not read graph file — it is not valid JSON.' });
        return;
      }
      // Pre-flight the file before touching the canvas, so a malformed / outdated /
      // unknown-node graph fails with a readable message instead of throwing
      // half-way through deserialization and leaving a broken graph behind.
      const issues = validateSerializedGraph(data, (t) => domain.nodeRegistry.get(t) !== undefined);
      if (issues.length > 0) {
        setStatus({ type: 'error', message: `Cannot load this graph:\n${issues.join('\n')}` });
        return;
      }
      graphRef.current = deserializeGraph(data);
      await runShape();
    },
    [runShape, domain],
  );

  // Save a subgraph node as a reusable block
  const saveBlock = useCallback(
    async (nodeId: string) => {
      const currentGraph = getCurrentGraph();
      const node = currentGraph.nodes.get(nodeId);
      if (!node || !node.subgraph) return;

      const blockData = {
        name: node.properties.blockName || 'Custom Block',
        description: `Custom block with ${node.subgraph.nodes.size} nodes`,
        subgraph: serializeGraphData(node.subgraph),
      };

      try {
        const res = await fetch(apiUrl('/blocks/save'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(blockData),
        });
        const data = await res.json();
        if (data.status === 'ok') {
          setStatus({ type: 'success', message: `Block "${blockData.name}" saved` });
          setTimeout(() => setStatus((s) => (s.type === 'success' ? { type: 'idle' } : s)), 3000);
          await refreshBlocks();
        }
      } catch {
        setStatus({ type: 'error', message: 'Failed to save block — is the backend running?' });
      }
    },
    [getCurrentGraph, refreshBlocks],
  );

  // Delete a saved block
  const deleteBlock = useCallback(
    async (filename: string) => {
      try {
        await fetch(apiUrl(`/blocks/${filename}`), { method: 'DELETE' });
        await refreshBlocks();
      } catch {
        /* ignore */
      }
    },
    [refreshBlocks],
  );

  // Add a saved block to the graph as a new subgraph node
  const addBlockFromTemplate = useCallback(
    async (filename: string, position: { x: number; y: number }) => {
      try {
        const res = await fetch(apiUrl(`/blocks/${filename}`));
        const data = await res.json();
        if (data.status !== 'ok') return;

        const block = data.block;
        const currentGraph = getCurrentGraph();
        const id = `subgraph.block-${Date.now()}`;
        const node = createNode(id, 'subgraph.block', position, { blockName: block.name });
        node.subgraph = deserializeGraphData(block.subgraph);
        addNode(currentGraph, node);
        invalidateModel();
        await runShape();
      } catch {
        setStatus({ type: 'error', message: 'Failed to load block' });
      }
    },
    [getCurrentGraph, invalidateModel, runShape],
  );

  // --- Backend Communication ---

  const runInfer = useCallback(async () => {
    if (!modelTrained) {
      setStatus({ type: 'error', message: 'No trained model — train first' });
      return;
    }
    if (modelStale) {
      setStatus({
        type: 'error',
        message: 'Model outdated — graph changed since last training. Retrain first.',
      });
      return;
    }
    setStatus({ type: 'running', message: 'Running inference...' });
    const r = await callBackend<{ results: any }>('/infer', serializeGraph(graphRef.current));
    if (!r.ok) {
      setStatus({ type: 'error', message: friendlyError(r.error) });
      return;
    }
    const result = r.data;

    // Apply results to nodes
    for (const [nodeId, nodeResult] of Object.entries(result.results.nodeResults) as [
      string,
      any,
    ][]) {
      const node = graphRef.current.nodes.get(nodeId);
      if (!node) continue;

      const existingMeta = node.lastResult?.metadata ?? {};
      node.lastResult = {
        outputs: nodeResult.outputs ?? {},
        metadata: {
          ...existingMeta,
          ...nodeResult.metadata,
          forwardResults: nodeResult.outputs,
        },
      };
    }

    syncToRF();

    const pred = result.results.prediction;
    if (pred) {
      setStatus({
        type: 'success',
        message: `Predicted: ${pred.predictedClass} (${(pred.confidence * 100).toFixed(1)}% confidence)`,
      });
    } else {
      setStatus({ type: 'success', message: 'Inference complete' });
    }
    tutorialEvent('infer-run');
    setTimeout(() => setStatus((s) => (s.type === 'success' ? { type: 'idle' } : s)), 5000);
  }, [syncToRF, modelTrained, modelStale]);

  // The shared TestResult type (dashboard/types) — the backend also sends the
  // optional confusionMatrix, which the dashboard and the agent tools read.
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  // True while a test evaluation is in flight — drives the dashboard's Test tab.
  const [testing, setTesting] = useState(false);

  // Returns the outcome as well as setting state, so a caller that awaits it
  // (the agent's run_test tool) can report the fresh result — its captured
  // props are a render-time snapshot and would still show the OLD testResult.
  const runTest = useCallback(async (): Promise<{ result: TestResult } | { error: string }> => {
    if (!modelTrained) {
      const error = 'No trained model — train first';
      setStatus({ type: 'error', message: error });
      return { error };
    }
    if (modelStale) {
      const error = 'Model outdated — graph changed since last training. Retrain first.';
      setStatus({ type: 'error', message: error });
      return { error };
    }
    // No toolbar status for test — results are shown in the dashboard test tab.
    setTesting(true);
    const r = await callBackend<{ result: TestResult }>('/evaluate-test', {
      graph: JSON.parse(saveGraph()),
    });
    setTesting(false);
    if (!r.ok) {
      const error = friendlyError(r.error);
      setStatus({ type: 'error', message: error });
      return { error };
    }
    setTestResult(r.data.result);
    return { result: r.data.result };
  }, [modelTrained, modelStale, saveGraph]);

  const [trainingProgress, setTrainingProgress] = useState<EpochData[]>([]);

  // Full history of per-epoch visualization snapshots (indexed by epoch number)
  const [snapshotHistory, setSnapshotHistory] = useState<Record<string, any>[]>([]);

  // Which epoch's snapshots to display (null = latest)
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);

  // Derived: the snapshot for the currently selected (or latest) epoch
  const liveSnapshots = (() => {
    if (snapshotHistory.length === 0) return {};
    const idx =
      selectedEpoch != null
        ? Math.min(selectedEpoch - 1, snapshotHistory.length - 1)
        : snapshotHistory.length - 1;
    return snapshotHistory[Math.max(0, idx)] ?? {};
  })();

  // Which nodes have their viz panel pinned open
  const [pinnedVizNodes, setPinnedVizNodes] = useState<Set<string>>(new Set());

  const toggleVizPin = useCallback((nodeId: string) => {
    setPinnedVizNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else {
        next.add(nodeId);
        tutorialEvent('viz-toggled');
      }
      return next;
    });
  }, []);

  const showAllViz = useCallback(() => {
    const g = getCurrentGraph();
    setPinnedVizNodes(new Set(g.nodes.keys()));
  }, [getCurrentGraph]);

  const hideAllViz = useCallback(() => {
    setPinnedVizNodes(new Set());
  }, []);

  // Batch-level progress within current epoch
  const [batchProgress, setBatchProgress] = useState<{
    batch: number;
    totalBatches: number;
  } | null>(null);
  // True from the moment training starts until it finishes/errors/cancels
  const [trainingActive, setTrainingActive] = useState(false);

  const exportPython = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/export-python'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(saveGraph()) }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Export failed' }));
        setStatus({ type: 'error', message: err.error ?? 'Export failed' });
        return;
      }
      const blob = await res.blob();
      const filename =
        (graphRef.current.name?.replace(/\s+/g, '_').toLowerCase() || 'model') + '.py';

      if ('showSaveFilePicker' in window) {
        try {
          const handle = await (window as any).showSaveFilePicker({
            suggestedName: filename,
            types: [
              {
                description: 'Python Script',
                accept: { 'text/x-python': ['.py'] },
              },
            ],
          });
          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
          setStatus({ type: 'success', message: 'Python code exported' });
        } catch {
          // User cancelled the picker
          return;
        }
      } else {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        setStatus({ type: 'success', message: 'Python code exported' });
      }
      setTimeout(() => setStatus((s) => (s.type === 'success' ? { type: 'idle' } : s)), 3000);
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, [saveGraph]);

  const flashSuccess = useCallback((message: string) => {
    setStatus({ type: 'success', message });
    setTimeout(() => setStatus((s) => (s.type === 'success' ? { type: 'idle' } : s)), 3000);
  }, []);

  // Save a self-contained model bundle (.ntmodel = graph + weights). Always reloads
  // correctly because the architecture travels with the weights. The frontend sends
  // the current graph; the backend pairs it with the trained weights.
  const saveModel = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/download-model'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(saveGraph()) }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Failed to download' }));
        setStatus({ type: 'error', message: err.error ?? 'No trained model to save' });
        return;
      }
      await downloadBlob(await res.blob(), 'model.ntmodel', 'NodeTorch Model', '.ntmodel');
      flashSuccess('Model saved');
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, [saveGraph, flashSuccess]);

  // Load a model bundle (.ntmodel): replaces the canvas graph AND restores weights.
  // The graph is embedded in the file, so the backend returns it for the canvas.
  const loadModel = useCallback(
    async (file: File) => {
      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(apiUrl('/upload-model'), { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'ok' && data.graph) {
          await loadGraph(JSON.stringify(data.graph));
          setModelTrained(true);
          setModelStale(false);
          flashSuccess(`Model loaded from "${file.name}"`);
        } else {
          setStatus({
            type: 'error',
            message: friendlyError(data.error ?? 'Failed to load model'),
          });
          setTimeout(() => setStatus((s) => (s.type === 'error' ? { type: 'idle' } : s)), 5000);
        }
      } catch {
        setStatus({ type: 'error', message: 'Cannot connect to backend' });
      }
    },
    [loadGraph, flashSuccess],
  );

  // Save just the trained weights (.pt). Loads back onto the CURRENT graph via
  // loadWeights — the architecture must match (use Save/Load Model to avoid that).
  const saveWeights = useCallback(async () => {
    try {
      const res = await fetch(apiUrl('/download-weights'));
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Failed to download' }));
        setStatus({ type: 'error', message: err.error ?? 'No trained model to save' });
        return;
      }
      await downloadBlob(await res.blob(), 'weights.pt', 'PyTorch Weights', '.pt');
      flashSuccess('Weights saved');
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, [flashSuccess]);

  // Load weights (.pt) onto the current graph. Sends the current graph so the
  // backend can rebuild the modules and pour the weights in (architecture must match).
  const loadWeights = useCallback(
    async (file: File) => {
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('graph', saveGraph());
        const res = await fetch(apiUrl('/upload-weights'), { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'ok') {
          setModelTrained(true);
          setModelStale(false);
          flashSuccess(`Weights loaded from "${file.name}"`);
        } else {
          setStatus({
            type: 'error',
            message: friendlyError(data.error ?? 'Failed to load weights'),
          });
          setTimeout(() => setStatus((s) => (s.type === 'error' ? { type: 'idle' } : s)), 5000);
          await runShape();
        }
      } catch {
        setStatus({ type: 'error', message: 'Cannot connect to backend' });
      }
    },
    [saveGraph, runShape, flashSuccess],
  );

  const trainWsRef = useRef<WebSocket | null>(null);

  const cancelTrain = useCallback(() => {
    const ws = trainWsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'cancel' }));
      setStatus({ type: 'error', message: 'Cancelling training...' });
    }
  }, []);

  const runTrain = useCallback(async () => {
    const errors = validateTraining(graphRef.current, domain.nodeRegistry);
    if (errors.length > 0) {
      setStatus({ type: 'error', message: errors.map((e) => e.message).join('\n') });
      return;
    }

    // No toolbar message during training — the dashboard is the single source
    // for progress/results. Keep type 'running' so a dropped WS is still detected.
    setStatus({ type: 'running' });
    setTrainingActive(true);
    tutorialEvent('training-started');
    setTrainingProgress([]);
    setSnapshotHistory([]);
    setSelectedEpoch(null);
    setBatchProgress(null);
    setTestResult(null); // retraining invalidates any previous test result
    const graphData = serializeGraph(graphRef.current);

    return new Promise<void>((resolve) => {
      const ws = new WebSocket(wsUrl('/ws'));
      trainWsRef.current = ws;

      function applyResults(results: any) {
        if (!results?.nodeResults) return;
        for (const [nodeId, nodeResult] of Object.entries(results.nodeResults) as [string, any][]) {
          const node = graphRef.current.nodes.get(nodeId);
          if (!node) continue;
          const existingMeta = node.lastResult?.metadata ?? {};
          node.lastResult = {
            outputs: nodeResult.outputs ?? {},
            metadata: {
              ...existingMeta,
              ...nodeResult.metadata,
              forwardResults: nodeResult.outputs,
            },
          };
        }
        syncToRF();
      }

      function cleanup() {
        trainWsRef.current = null;
        setTrainingActive(false);
        resolve();
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'batch') {
          setBatchProgress({ batch: msg.batch, totalBatches: msg.totalBatches });
          return;
        }

        if (msg.type === 'epoch') {
          setBatchProgress(null); // reset batch progress when epoch completes
          setTrainingProgress((prev) => [
            ...prev,
            {
              epoch: msg.epoch,
              totalEpochs: msg.totalEpochs,
              loss: msg.loss,
              accuracy: msg.accuracy,
              valLoss: msg.valLoss,
              valAccuracy: msg.valAccuracy,
              learningRate: msg.learningRate,
              time: msg.time,
              batches: msg.batches,
              samples: msg.samples,
              gradientFlow: msg.gradientFlow,
              perClassAccuracy: msg.perClassAccuracy,
              trackedSamples: msg.trackedSamples,
              generatedSamples: msg.generatedSamples,
              dLoss: msg.dLoss,
              gLoss: msg.gLoss,
              trainingMode: msg.trainingMode,
              perplexity: msg.perplexity,
              valPerplexity: msg.valPerplexity,
              generatedText: msg.generatedText,
            },
          ]);
          // Accumulate visualization snapshots per epoch
          if (msg.nodeSnapshots) {
            setSnapshotHistory((prev) => [...prev, msg.nodeSnapshots]);
          }
          // Per-epoch progress is shown in the dashboard, not the toolbar.
        }

        if (msg.type === 'train_result') {
          if (msg.status === 'ok') {
            applyResults(msg.results);
            setModelTrained(true);
            setModelStale(false);
            // Final result lives in the dashboard; clear the toolbar status
            // (also moves type off 'running' so onclose won't flag a lost WS).
            setStatus({ type: 'idle' });
          } else {
            setStatus({ type: 'error', message: friendlyError(msg.error ?? 'Training failed') });
          }
          ws.close();
          cleanup();
        }

        if (msg.type === 'cancelled') {
          setStatus({ type: 'running', message: 'Cancelling after current epoch...' });
        }
      };

      ws.onerror = () => {
        setStatus({ type: 'error', message: 'Cannot connect to backend — is the server running?' });
        cleanup();
      };

      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'train', graph: graphData }));
      };

      ws.onclose = () => {
        setStatus((s) =>
          s.type === 'running' ? { type: 'error', message: 'Connection lost during training' } : s,
        );
        cleanup();
      };
    });
  }, [domain, syncToRF]);

  // Handle React Flow's node changes (drag, select, remove)
  const onNodesChange = useCallback(
    (changes: RF.NodeChange[]) => {
      const g = getCurrentGraph();
      for (const change of changes) {
        if (change.type === 'position' && change.position) {
          const node = g.nodes.get(change.id);
          if (node) {
            node.position = change.position;
          }
        }
        if (change.type === 'remove') {
          snapshot();
          g.nodes.delete(change.id);
          g.edges = g.edges.filter(
            (e) => e.source.nodeId !== change.id && e.target.nodeId !== change.id,
          );
          invalidateModel();
        }
      }
      setRFNodes((nds) => RF.applyNodeChanges(changes, nds));
    },
    [invalidateModel, getCurrentGraph, snapshot],
  );

  const onEdgesChange = useCallback(
    (changes: RF.EdgeChange[]) => {
      const g = getCurrentGraph();
      for (const change of changes) {
        if (change.type === 'remove') {
          snapshot();
          const edge = g.edges.find((e) => e.id === change.id);
          if (edge) {
            markDirty(g, edge.target.nodeId);
          }
          const idx = g.edges.findIndex((e) => e.id === change.id);
          if (idx !== -1) g.edges.splice(idx, 1);
          invalidateModel();
        }
      }
      setRFEdges((eds) => RF.applyEdgeChanges(changes, eds));
    },
    [invalidateModel, getCurrentGraph, snapshot],
  );

  // --- Auto-organize: space nodes evenly along the topological order ---
  const organizeGraph = useCallback(() => {
    const g = getCurrentGraph();
    if (g.nodes.size === 0) return;
    const positions = computeLayout(g);
    snapshot();
    for (const [id, pos] of positions) {
      const node = g.nodes.get(id);
      if (node) node.position = pos;
    }
    syncToRF();
  }, [getCurrentGraph, syncToRF, snapshot]);

  // --- Copy/Paste ---
  const clipboard = useRef<{ nodes: any[]; edges: any[] } | null>(null);

  const copySelected = useCallback(() => {
    const currentGraph = getCurrentGraph();
    const selectedIds = new Set(rfNodes.filter((n) => n.selected).map((n) => n.id));
    if (selectedIds.size === 0) return;

    const copiedNodes = Array.from(currentGraph.nodes.values())
      .filter((n) => selectedIds.has(n.id))
      .map((n) => ({
        type: n.type,
        position: { ...n.position },
        properties: { ...n.properties },
        subgraph: n.subgraph ? serializeGraphData(n.subgraph) : undefined,
      }));

    // Copy edges that connect only between selected nodes
    const copiedEdges = currentGraph.edges
      .filter((e) => selectedIds.has(e.source.nodeId) && selectedIds.has(e.target.nodeId))
      .map((e) => ({
        sourceIdx: Array.from(selectedIds).indexOf(e.source.nodeId),
        sourcePort: e.source.portId,
        targetIdx: Array.from(selectedIds).indexOf(e.target.nodeId),
        targetPort: e.target.portId,
      }));

    clipboard.current = { nodes: copiedNodes, edges: copiedEdges };
  }, [getCurrentGraph, rfNodes]);

  const paste = useCallback(async () => {
    if (!clipboard.current || clipboard.current.nodes.length === 0) return;

    snapshot();
    const currentGraph = getCurrentGraph();
    const OFFSET = 50;
    const newIds: string[] = [];

    // Create new nodes with offset positions
    for (const copied of clipboard.current.nodes) {
      const id = `${copied.type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const node = createNode(
        id,
        copied.type,
        {
          x: copied.position.x + OFFSET,
          y: copied.position.y + OFFSET,
        },
        { ...copied.properties },
      );

      if (copied.subgraph) {
        node.subgraph = deserializeGraphData(copied.subgraph);
      }

      addNode(currentGraph, node);
      newIds.push(id);
    }

    // Recreate edges between pasted nodes
    for (const ce of clipboard.current.edges) {
      if (
        ce.sourceIdx >= 0 &&
        ce.targetIdx >= 0 &&
        ce.sourceIdx < newIds.length &&
        ce.targetIdx < newIds.length
      ) {
        const edgeId = `e-${edgeCounter++}`;
        const edge = createEdge(
          edgeId,
          newIds[ce.sourceIdx],
          ce.sourcePort,
          newIds[ce.targetIdx],
          ce.targetPort,
        );
        try {
          addGraphEdge(currentGraph, edge);
        } catch {
          /* skip invalid edges */
        }
      }
    }

    invalidateModel();
    await runShape();
  }, [getCurrentGraph, snapshot, invalidateModel, runShape]);

  // Resolve snapshots for current navigation depth.
  // At root: use top-level snapshots directly.
  // Inside a subgraph: extract innerSnapshots from the parent subgraph node.
  // Checks both live training snapshots and forward pass metadata.
  const resolvedSnapshots = (() => {
    if (navStack.length === 0) return liveSnapshots;

    // Try live training snapshots first
    let snaps = liveSnapshots;
    let found = true;
    for (const entry of navStack) {
      const parentSnap = snaps[entry.nodeId];
      if (parentSnap?.innerSnapshots) {
        snaps = parentSnap.innerSnapshots;
      } else {
        found = false;
        break;
      }
    }
    if (found && Object.keys(snaps).length > 0) return snaps;

    // Fallback: check forward pass metadata on the subgraph node
    let g = graphRef.current;
    for (const entry of navStack) {
      const node = g.nodes.get(entry.nodeId);
      if (!node) return {};
      const innerSnaps = node.lastResult?.metadata?.innerSnapshots;
      if (innerSnaps) return innerSnaps;
      if (node.subgraph) g = node.subgraph;
    }
    return {};
  })();

  // Live forward-validation, keyed by nodeId, for per-node "needs input" badges.
  // Recomputed on every mutation (graphVersion) or subgraph navigation.
  const validationErrors = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const err of validateForward(getCurrentGraph(), domain.nodeRegistry)) {
      if (!err.nodeId) continue;
      const list = map.get(err.nodeId);
      if (list) list.push(err.message);
      else map.set(err.nodeId, [err.message]);
    }
    return map;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphVersion, navStack, domain]);

  return {
    graph: graphRef.current,
    currentGraph: getCurrentGraph(),
    graphVersion,
    validationErrors,
    rfNodes,
    rfEdges,
    navStack,
    enterSubgraph,
    navigateTo,
    exitSubgraph,
    onNodesChange,
    onEdgesChange,
    addNode: addNodeToGraph,
    removeNode: removeNodeFromGraph,
    connect: connectNodes,
    updateProperty,
    removeEdge: removeEdgeByEndpoints,
    getCurrentGraph,
    runShape,
    isValidConnection,
    beginBatch,
    endBatch,
    saveGraph,
    loadGraph,
    clearGraph,
    runInfer,
    runTest,
    testResult,
    testing,
    runTrain,
    cancelTrain,
    saveModel,
    loadModel,
    saveWeights,
    loadWeights,
    status,
    modelTrained,
    modelStale,
    connectionError,
    trainingProgress,
    batchProgress,
    trainingActive,
    undo,
    redo,
    copySelected,
    paste,
    savedBlocks,
    saveBlock,
    deleteBlock,
    addBlockFromTemplate,
    organizeGraph,
    liveSnapshots: resolvedSnapshots,
    snapshotHistory,
    selectedEpoch,
    setSelectedEpoch,
    exportPython,
    pinnedVizNodes,
    toggleVizPin,
    showAllViz,
    hideAllViz,
  };
}
