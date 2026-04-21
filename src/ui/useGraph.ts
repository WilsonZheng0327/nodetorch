// Bridge between our engine (Layers 1-5) and React Flow's state format.
// Owns the Graph, runs the engine, and converts to/from React Flow nodes/edges.

import { useState, useCallback, useRef, useEffect } from 'react';
import * as RF from '@xyflow/react';
import { tutorialEvent } from './tutorial/TutorialPanel';
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

// --- Serialization ---

interface SerializedNode {
  id: string;
  type: string;
  position: { x: number; y: number };
  properties: Record<string, any>;
  subgraph?: SerializedGraphData;
}

interface SerializedGraphData {
  id: string;
  name: string;
  nodes: SerializedNode[];
  edges: { id: string; source: { nodeId: string; portId: string }; target: { nodeId: string; portId: string } }[];
}

interface SerializedGraph {
  version: '1.0';
  graph: SerializedGraphData;
}

function serializeGraphData(graph: Graph): SerializedGraphData {
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

function serializeGraph(graph: Graph): SerializedGraph {
  return { version: '1.0', graph: serializeGraphData(graph) };
}

function deserializeGraphData(data: SerializedGraphData): Graph {
  const graph = createGraph(data.id, data.name);
  for (const n of data.nodes) {
    const node = createNode(n.id, n.type, n.position, n.properties);
    if (n.subgraph) {
      node.subgraph = deserializeGraphData(n.subgraph);
    }
    addNode(graph, node);
  }
  for (const e of data.edges) {
    addGraphEdge(graph, { id: e.id, source: e.source, target: e.target });
  }
  return graph;
}

function deserializeGraph(data: SerializedGraph): Graph {
  return deserializeGraphData(data.graph);
}

// --- Friendly error translation ---

/** Translate common PyTorch errors into student-friendly messages. */
function friendlyError(msg: string): string {
  if (msg.includes('mat1 and mat2 shapes cannot be multiplied')) {
    const match = msg.match(/(\d+x\d+).*?(\d+x\d+)/);
    if (match) return `Shape mismatch in Linear layer: input is ${match[1]} but weights expect ${match[2]}. Check upstream layer output size.`;
  }
  if (msg.includes('Expected 4-dimensional input')) return 'This layer expects a 4D tensor [B,C,H,W]. Add a Reshape node or check connections.';
  if (msg.includes('Expected 3-dimensional input')) return 'This layer expects a 3D tensor [B,seq,features]. Check input dimensions.';
  if (msg.includes('Expected 2-dimensional input')) return 'This layer expects a 2D tensor [B,features]. Did you forget a Flatten layer?';
  if (msg.includes('size mismatch')) return `Tensor size mismatch — shapes don't align. Check that connected layers have compatible dimensions.`;
  if (msg.includes('CUDA out of memory')) return 'GPU out of memory. Try reducing batch size or using CPU.';
  if (msg.includes('is not a valid device')) return 'Selected device not available. Switch to CPU in the dashboard System tab.';
  if (msg.includes('negative dimension')) return 'Layer produced a negative dimension — kernel/stride/padding combination is too large for the input size.';
  return msg;
}

// --- The hook ---

let edgeCounter = Date.now();

export function useGraph(domain: DomainContext) {
  const graphRef = useRef<Graph>(createGraph('main', 'Main Graph'));
  const [rfNodes, setRFNodes] = useState<RF.Node[]>([]);
  const [rfEdges, setRFEdges] = useState<RF.Edge[]>([]);
  const [graphVersion, setGraphVersion] = useState(0);  // bumped on every mutation to force re-renders
  const [status, setStatus] = useState<{ type: 'idle' | 'running' | 'success' | 'error'; message?: string }>({ type: 'idle' });
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [modelTrained, setModelTrained] = useState(false);
  const [modelStale, setModelStale] = useState(false);

  // --- Undo/Redo (refs + snapshot here, undo/redo functions after syncToRF) ---
  const undoStack = useRef<string[]>([]);
  const redoStack = useRef<string[]>([]);
  const isUndoRedo = useRef(false);

  const snapshot = useCallback(() => {
    if (isUndoRedo.current) return;
    undoStack.current.push(JSON.stringify(serializeGraphData(graphRef.current)));
    redoStack.current = [];
    if (undoStack.current.length > 50) undoStack.current.shift();
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
      const res = await fetch('http://localhost:8000/blocks');
      const data = await res.json();
      if (data.status === 'ok') setSavedBlocks(data.blocks);
    } catch { /* backend not running */ }
  }, []);

  // Load on mount
  useEffect(() => { refreshBlocks(); }, [refreshBlocks]);

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
  const enterSubgraph = useCallback((nodeId: string) => {
    const currentGraph = getCurrentGraph();
    const node = currentGraph.nodes.get(nodeId);
    if (!node || node.type !== 'subgraph.block' || !node.subgraph) return;

    const newStack: NavEntry[] = [
      ...navStack,
      { graphId: node.subgraph.id, label: node.properties.blockName || 'Block', nodeId },
    ];
    setNavStack(newStack);
    syncToGraph(resolveGraph(newStack));
  }, [getCurrentGraph, navStack, resolveGraph, syncToGraph]);

  // Go back to a specific level in the breadcrumb
  const navigateTo = useCallback((depth: number) => {
    const newStack = navStack.slice(0, depth);
    setNavStack(newStack);
    syncToGraph(resolveGraph(newStack));
  }, [navStack, resolveGraph, syncToGraph]);

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
        console.warn(`[nodetorch] Pruned invalid edge ${e.id}: ${e.source.portId} → ${e.target.portId}`);
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
      await domain.engine.execute(
        graphRef.current,
        'shape',
        (nodeType, executorKey) => domain.nodeRegistry.getExecutor(nodeType, executorKey),
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
      if (node?.subgraph) { g = node.subgraph; validDepth++; } else break;
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
      if (node?.subgraph) { g = node.subgraph; validDepth++; } else break;
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

      if (!sourcePort) return reject(`source port "${connection.sourceHandle}" not found on ${sourceName}`);
      if (!targetPort) return reject(`target port "${connection.targetHandle}" not found on ${targetName}`);

      // Rule 3: data type compatibility
      if (!domain.typeRegistry.isCompatible(sourcePort.dataType, targetPort.dataType)) {
        return reject(`type mismatch: ${sourcePort.dataType} → ${targetPort.dataType}`);
      }

      // Rule 4: check allowMultiple on target port
      if (!targetPort.allowMultiple) {
        const alreadyConnected = graph.edges.some(
          (e) =>
            e.target.nodeId === connection.target &&
            e.target.portId === connection.targetHandle,
        );
        if (alreadyConnected) return reject(`port "${targetPort.name}" already connected`);
      }

      return true;
    },
    [domain, getCurrentGraph],
  );

  // --- Actions the UI can trigger ---

  const addNodeToGraph = useCallback(
    async (type: string, position: { x: number; y: number }) => {
      const def = domain.nodeRegistry.get(type);
      if (!def) return;

      // Build default properties from the definition
      const properties: Record<string, any> = {};
      for (const prop of def.getProperties()) {
        properties[prop.id] = prop.defaultValue;
      }

      const id = `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const node = createNode(id, type, position, properties);

      // If this is a subgraph block, create the inner graph with default sentinels
      if (type === 'subgraph.block') {
        const innerGraph = createGraph(`${id}-inner`, 'Inner Graph');
        const inputNode = createNode('input', 'subgraph.input', { x: 0, y: 100 }, { portCount: 1, portNames: 'in' });
        const outputNode = createNode('output', 'subgraph.output', { x: 400, y: 100 }, { portCount: 1, portNames: 'out' });
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
      if (type.startsWith('ml.layers.maxpool') || type.startsWith('ml.layers.avgpool')) tutorialEvent('node-added-pool');
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
      const data: SerializedGraph = JSON.parse(json);
      graphRef.current = deserializeGraph(data);
      await runShape();
    },
    [runShape],
  );

  // Save a subgraph node as a reusable block
  const saveBlock = useCallback(async (nodeId: string) => {
    const currentGraph = getCurrentGraph();
    const node = currentGraph.nodes.get(nodeId);
    if (!node || !node.subgraph) return;

    const blockData = {
      name: node.properties.blockName || 'Custom Block',
      description: `Custom block with ${node.subgraph.nodes.size} nodes`,
      subgraph: serializeGraphData(node.subgraph),
    };

    try {
      const res = await fetch('http://localhost:8000/blocks/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(blockData),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setStatus({ type: 'success', message: `Block "${blockData.name}" saved` });
        setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 3000);
        await refreshBlocks();
      }
    } catch {
      setStatus({ type: 'error', message: 'Failed to save block — is the backend running?' });
    }
  }, [getCurrentGraph, refreshBlocks]);

  // Delete a saved block
  const deleteBlock = useCallback(async (filename: string) => {
    try {
      await fetch(`http://localhost:8000/blocks/${filename}`, { method: 'DELETE' });
      await refreshBlocks();
    } catch { /* ignore */ }
  }, [refreshBlocks]);

  // Add a saved block to the graph as a new subgraph node
  const addBlockFromTemplate = useCallback(async (filename: string, position: { x: number; y: number }) => {
    try {
      const res = await fetch(`http://localhost:8000/blocks/${filename}`);
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
  }, [getCurrentGraph, invalidateModel, runShape]);

  // --- Backend Communication ---

  const runForward = useCallback(async () => {
    const errors = validateForward(graphRef.current, domain.nodeRegistry);
    if (errors.length > 0) {
      setStatus({ type: 'error', message: errors.map((e) => e.message).join('\n') });
      return;
    }

    setStatus({ type: 'running' });
    const graphData = serializeGraph(graphRef.current);

    let response: Response;
    try {
      response = await fetch('http://localhost:8000/forward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphData),
      });
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend — is the server running?' });
      return;
    }

    let result: any;
    try {
      result = await response.json();
    } catch {
      setStatus({ type: 'error', message: `Backend error (HTTP ${response.status})` });
      return;
    }

    if (result.status !== 'ok') {
      setStatus({ type: 'error', message: friendlyError(result.error ?? result.detail ?? 'Forward pass failed') });
      return;
    }

    // Check if any nodes reported errors
    let hasErrors = false;
    for (const [nodeId, nodeResult] of Object.entries(result.results) as [string, any][]) {
      const node = graphRef.current.nodes.get(nodeId);
      if (!node) continue;

      if (nodeResult.metadata?.error) {
        hasErrors = true;
      }

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
    setStatus(hasErrors
      ? { type: 'error', message: 'Forward pass completed with errors — check nodes' }
      : { type: 'success', message: 'Forward pass complete' },
    );
    if (!hasErrors) tutorialEvent('forward-run');

    // Clear success status after a few seconds
    if (!hasErrors) {
      setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 3000);
    }
  }, [domain, syncToRF]);

  const runInfer = useCallback(async () => {
    if (!modelTrained) {
      setStatus({ type: 'error', message: 'No trained model — train first' });
      return;
    }
    if (modelStale) {
      setStatus({ type: 'error', message: 'Model outdated — graph changed since last training. Retrain first.' });
      return;
    }
    setStatus({ type: 'running', message: 'Running inference...' });
    const graphData = serializeGraph(graphRef.current);

    let response: Response;
    try {
      response = await fetch('http://localhost:8000/infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphData),
      });
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend — is the server running?' });
      return;
    }

    let result: any;
    try {
      result = await response.json();
    } catch {
      setStatus({ type: 'error', message: `Backend error (HTTP ${response.status})` });
      return;
    }

    if (result.status !== 'ok') {
      setStatus({ type: 'error', message: friendlyError(result.error ?? 'Inference failed') });
      return;
    }

    // Apply results to nodes
    for (const [nodeId, nodeResult] of Object.entries(result.results.nodeResults) as [string, any][]) {
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
    setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 5000);
  }, [syncToRF, modelTrained, modelStale]);

  const [testResult, setTestResult] = useState<{
    testLoss: number; testAccuracy: number; testSamples: number;
    perClassAccuracy: { cls: number; name: string; accuracy: number; count: number }[];
  } | null>(null);

  const runTest = useCallback(async () => {
    if (!modelTrained) {
      setStatus({ type: 'error', message: 'No trained model — train first' });
      return;
    }
    setStatus({ type: 'running', message: 'Evaluating on test set...' });
    try {
      const res = await fetch('http://localhost:8000/evaluate-test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(saveGraph()) }),
      });
      const data = await res.json();
      if (data.status !== 'ok') {
        setStatus({ type: 'error', message: friendlyError(data.error ?? 'Test evaluation failed') });
        return;
      }
      setTestResult(data.result);
      const acc = (data.result.testAccuracy * 100).toFixed(1);
      const loss = data.result.testLoss.toFixed(4);
      setStatus({ type: 'success', message: `Test set: ${acc}% accuracy, ${loss} loss (${data.result.testSamples} samples)` });
      setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 8000);
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, [modelTrained, saveGraph]);

  const [trainingProgress, setTrainingProgress] = useState<{ epoch: number; loss: number; accuracy: number }[]>([]);

  // Full history of per-epoch visualization snapshots (indexed by epoch number)
  const [snapshotHistory, setSnapshotHistory] = useState<Record<string, any>[]>([]);

  // Which epoch's snapshots to display (null = latest)
  const [selectedEpoch, setSelectedEpoch] = useState<number | null>(null);

  // Derived: the snapshot for the currently selected (or latest) epoch
  const liveSnapshots = (() => {
    if (snapshotHistory.length === 0) return {};
    const idx = selectedEpoch != null
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
      else { next.add(nodeId); tutorialEvent('viz-toggled'); }
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
  const [batchProgress, setBatchProgress] = useState<{ batch: number; totalBatches: number } | null>(null);
  // True from the moment training starts until it finishes/errors/cancels
  const [trainingActive, setTrainingActive] = useState(false);

  // Backprop animation: map of nodeId -> { delayMs, intensity }
  const [backpropAnim, setBackpropAnim] = useState<Record<string, { delayMs: number; intensity: number }> | null>(null);

  const simulateBackprop = useCallback(async () => {
    setStatus({ type: 'running', message: 'Simulating backprop...' });
    try {
      const res = await fetch('http://localhost:8000/simulate-backprop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(saveGraph()) }),
      });
      const data = await res.json();
      if (data.status !== 'ok') {
        setStatus({ type: 'error', message: data.error ?? 'Backprop simulation failed' });
        return;
      }
      const flow: { nodeId: string; norm: number }[] = data.result.flow;
      // Reverse: animate from last layer backward to first
      const reversed = [...flow].reverse();
      const maxNorm = Math.max(...reversed.map((f) => f.norm), 1e-8);
      const stepMs = 150;
      const anim: Record<string, { delayMs: number; intensity: number }> = {};
      reversed.forEach((f, i) => {
        anim[f.nodeId] = {
          delayMs: i * stepMs,
          intensity: Math.min(1, 0.3 + (f.norm / maxNorm) * 0.7),
        };
      });
      setBackpropAnim(anim);
      setStatus({ type: 'success', message: `Backprop: loss=${data.result.loss?.toFixed(4)}` });
      const totalDuration = reversed.length * stepMs + 800;
      setTimeout(() => {
        setBackpropAnim(null);
        setStatus((s) => s.type === 'success' ? { type: 'idle' } : s);
      }, totalDuration);
    } catch (e) {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, []);

  const exportPython = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/export-python', {
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
      const filename = (graphRef.current.name?.replace(/\s+/g, '_').toLowerCase() || 'model') + '.py';

      if ('showSaveFilePicker' in window) {
        try {
          const handle = await (window as any).showSaveFilePicker({
            suggestedName: filename,
            types: [{
              description: 'Python Script',
              accept: { 'text/x-python': ['.py'] },
            }],
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
      setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 3000);
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, [saveGraph]);

  const saveModel = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/download-weights');
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Failed to download' }));
        setStatus({ type: 'error', message: err.error ?? 'No trained model to save' });
        return;
      }
      const blob = await res.blob();

      // Use File System Access API if available
      if ('showSaveFilePicker' in window) {
        try {
          const handle = await (window as any).showSaveFilePicker({
            suggestedName: 'weights.pt',
            types: [{ description: 'PyTorch Weights', accept: { 'application/octet-stream': ['.pt'] } }],
          });
          const writable = await handle.createWritable();
          await writable.write(blob);
          await writable.close();
          setStatus({ type: 'success', message: 'Weights saved' });
          setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 3000);
          return;
        } catch {
          return; // user cancelled
        }
      }

      // Fallback: auto-download
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'weights.pt';
      a.click();
      URL.revokeObjectURL(url);
      setStatus({ type: 'success', message: 'Weights saved' });
      setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 3000);
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, []);

  const weightsInputRef = useRef<HTMLInputElement>(null);
  const loadModel = useCallback(async () => {
    weightsInputRef.current?.click();
  }, []);

  const handleWeightsFile = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('graph', saveGraph());
      const res = await fetch('http://localhost:8000/upload-weights', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.status === 'ok') {
        setModelTrained(true);
        setModelStale(false);
        setStatus({ type: 'success', message: `Weights loaded from "${file.name}"` });
        setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 3000);
      } else {
        setStatus({ type: 'error', message: friendlyError(data.error ?? 'Failed to load weights') });
        setTimeout(() => setStatus((s) => s.type === 'error' ? { type: 'idle' } : s), 5000);
        await runShape();
      }
    } catch {
      setStatus({ type: 'error', message: 'Cannot connect to backend' });
    }
  }, [saveGraph, runShape]);

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

    setStatus({ type: 'running', message: 'Training...' });
    setTrainingActive(true);
    tutorialEvent('training-started');
    setTrainingProgress([]);
    setSnapshotHistory([]);
    setSelectedEpoch(null);
    setBatchProgress(null);
    const graphData = serializeGraph(graphRef.current);

    return new Promise<void>((resolve) => {
      const ws = new WebSocket('ws://localhost:8000/ws');
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
          setTrainingProgress((prev) => [...prev, {
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
          }]);
          // Accumulate visualization snapshots per epoch
          if (msg.nodeSnapshots) {
            setSnapshotHistory((prev) => [...prev, msg.nodeSnapshots]);
          }
          const progressStr = msg.totalEpochs ? ` [${msg.epoch}/${msg.totalEpochs}]` : '';
          const timeStr = msg.time != null ? ` (${msg.time}s)` : '';
          setStatus({
            type: 'running',
            message: `Epoch${progressStr} — loss: ${msg.loss?.toFixed(4)}, acc: ${(msg.accuracy * 100)?.toFixed(1)}%${timeStr}`,
          });
        }

        if (msg.type === 'train_result') {
          if (msg.status === 'ok') {
            applyResults(msg.results);
            setModelTrained(true);
            setModelStale(false);
            const message = msg.cancelled ? 'Training cancelled' : 'Training complete';
            setStatus({ type: 'success', message });
            setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 5000);
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
        setStatus((s) => s.type === 'running'
          ? { type: 'error', message: 'Connection lost during training' }
          : s,
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

    // Build adjacency for a simple left-to-right layout
    // Assign each node a "column" based on longest path from a root
    const inDegree = new Map<string, number>();
    const adj = new Map<string, string[]>();
    for (const node of g.nodes.values()) {
      inDegree.set(node.id, 0);
      adj.set(node.id, []);
    }
    for (const edge of g.edges) {
      const prev = inDegree.get(edge.target.nodeId) ?? 0;
      inDegree.set(edge.target.nodeId, prev + 1);
      adj.get(edge.source.nodeId)?.push(edge.target.nodeId);
    }

    // Longest-path layering (ensures connected nodes are in adjacent columns)
    const depth = new Map<string, number>();
    const queue: string[] = [];
    for (const [id, deg] of inDegree) {
      if (deg === 0) { depth.set(id, 0); queue.push(id); }
    }
    // Also handle disconnected nodes
    if (queue.length === 0) {
      for (const id of g.nodes.keys()) { depth.set(id, 0); queue.push(id); }
    }

    let maxDepth = 0;
    while (queue.length > 0) {
      const id = queue.shift()!;
      const d = depth.get(id) ?? 0;
      for (const tgt of adj.get(id) ?? []) {
        const newD = d + 1;
        if (newD > (depth.get(tgt) ?? 0)) {
          depth.set(tgt, newD);
          if (newD > maxDepth) maxDepth = newD;
        }
        const remaining = (inDegree.get(tgt) ?? 1) - 1;
        inDegree.set(tgt, remaining);
        if (remaining === 0) queue.push(tgt);
      }
    }

    // Assign unvisited nodes (disconnected) to column 0
    for (const id of g.nodes.keys()) {
      if (!depth.has(id)) depth.set(id, 0);
    }

    // Group by column
    const columns = new Map<number, string[]>();
    for (const [id, d] of depth) {
      if (!columns.has(d)) columns.set(d, []);
      columns.get(d)!.push(id);
    }

    // Layout with spacing
    const COL_GAP = 250;
    const ROW_GAP = 120;
    snapshot();
    for (const [col, ids] of columns) {
      // Sort nodes within column by their current Y to preserve relative order
      ids.sort((a, b) => (g.nodes.get(a)!.position.y - g.nodes.get(b)!.position.y));
      const totalHeight = (ids.length - 1) * ROW_GAP;
      const startY = -totalHeight / 2;
      for (let i = 0; i < ids.length; i++) {
        const node = g.nodes.get(ids[i])!;
        node.position = { x: col * COL_GAP, y: startY + i * ROW_GAP };
      }
    }
    syncToRF();
  }, [getCurrentGraph, syncToRF, snapshot]);

  // --- Copy/Paste ---
  const clipboard = useRef<{ nodes: any[]; edges: any[] } | null>(null);

  const copySelected = useCallback(() => {
    const currentGraph = getCurrentGraph();
    const selectedIds = new Set(
      rfNodes.filter((n) => n.selected).map((n) => n.id),
    );
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
      const node = createNode(id, copied.type, {
        x: copied.position.x + OFFSET,
        y: copied.position.y + OFFSET,
      }, { ...copied.properties });

      if (copied.subgraph) {
        node.subgraph = deserializeGraphData(copied.subgraph);
      }

      addNode(currentGraph, node);
      newIds.push(id);
    }

    // Recreate edges between pasted nodes
    for (const ce of clipboard.current.edges) {
      if (ce.sourceIdx >= 0 && ce.targetIdx >= 0 && ce.sourceIdx < newIds.length && ce.targetIdx < newIds.length) {
        const edgeId = `e-${edgeCounter++}`;
        const edge = createEdge(edgeId, newIds[ce.sourceIdx], ce.sourcePort, newIds[ce.targetIdx], ce.targetPort);
        try {
          addGraphEdge(currentGraph, edge);
        } catch { /* skip invalid edges */ }
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

  return {
    graph: graphRef.current,
    currentGraph: getCurrentGraph(),
    graphVersion,
    rfNodes,
    rfEdges,
    navStack,
    enterSubgraph,
    navigateTo,
    onNodesChange,
    onEdgesChange,
    addNode: addNodeToGraph,
    removeNode: removeNodeFromGraph,
    connect: connectNodes,
    updateProperty,
    runShape,
    isValidConnection,
    saveGraph,
    loadGraph,
    clearGraph,
    runForward,
    runInfer,
    runTest,
    testResult,
    runTrain,
    cancelTrain,
    saveModel,
    loadModel,
    weightsInputRef,
    handleWeightsFile,
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
    backpropAnim,
    simulateBackprop,
    exportPython,
    pinnedVizNodes,
    toggleVizPin,
    showAllViz,
    hideAllViz,
  };
}
