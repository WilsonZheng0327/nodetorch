// Bridge between our engine (Layers 1-5) and React Flow's state format.
// Owns the Graph, runs the engine, and converts to/from React Flow nodes/edges.

import { useState, useCallback, useRef, useEffect } from 'react';
import * as RF from '@xyflow/react';
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

// --- The hook ---

let edgeCounter = Date.now();

export function useGraph(domain: DomainContext) {
  const graphRef = useRef<Graph>(createGraph('main', 'Main Graph'));
  const [rfNodes, setRFNodes] = useState<RF.Node[]>([]);
  const [rfEdges, setRFEdges] = useState<RF.Edge[]>([]);
  const [status, setStatus] = useState<{ type: 'idle' | 'running' | 'success' | 'error'; message?: string }>({ type: 'idle' });
  const [modelTrained, setModelTrained] = useState(false);
  const [modelStale, setModelStale] = useState(false);

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

  // Enter a subgraph node (double-click)
  const enterSubgraph = useCallback((nodeId: string) => {
    const currentGraph = getCurrentGraph();
    const node = currentGraph.nodes.get(nodeId);
    if (!node || node.type !== 'subgraph.block' || !node.subgraph) return;

    setNavStack((prev) => [
      ...prev,
      {
        graphId: node.subgraph!.id,
        label: node.properties.blockName || 'Block',
        nodeId,
      },
    ]);
  }, [getCurrentGraph]);

  // Go back to a specific level in the breadcrumb
  const navigateTo = useCallback((depth: number) => {
    setNavStack((prev) => prev.slice(0, depth));
  }, []);

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
  }, [pruneInvalidEdges, getCurrentGraph]);

  // Run shape inference on the graph, then sync to React Flow
  const runShape = useCallback(async () => {
    await domain.engine.execute(
      graphRef.current,
      'shape',
      (nodeType, executorKey) => domain.nodeRegistry.getExecutor(nodeType, executorKey),
    );
    syncToRF();
  }, [domain, syncToRF]);

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

      const id = `${type}-${Date.now()}`;
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

      const currentGraph = getCurrentGraph();
      addNode(currentGraph, node);
      invalidateModel();
      await runShape();
    },
    [domain, runShape, invalidateModel, getCurrentGraph],
  );

  const removeNodeFromGraph = useCallback(
    async (nodeId: string) => {
      removeNode(getCurrentGraph(), nodeId);
      invalidateModel();
      await runShape();
    },
    [runShape, invalidateModel, getCurrentGraph],
  );

  const connectNodes = useCallback(
    async (connection: RF.Connection) => {
      if (!connection.sourceHandle || !connection.targetHandle) {
        console.warn('[nodetorch] Connection missing handle ids', connection);
        return;
      }
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
    },
    [runShape, invalidateModel, getCurrentGraph],
  );

  const updateProperty = useCallback(
    async (nodeId: string, key: string, value: any) => {
      setProperty(getCurrentGraph(), nodeId, key, value);
      invalidateModel();
      await runShape();
    },
    [runShape, invalidateModel],
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
      setStatus({ type: 'error', message: result.error ?? result.detail ?? 'Forward pass failed' });
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
      setStatus({ type: 'error', message: result.error ?? 'Inference failed' });
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
    setTimeout(() => setStatus((s) => s.type === 'success' ? { type: 'idle' } : s), 5000);
  }, [syncToRF, modelTrained, modelStale]);

  const [trainingProgress, setTrainingProgress] = useState<{ epoch: number; loss: number; accuracy: number }[]>([]);

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
    setTrainingProgress([]);
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
        resolve();
      }

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === 'epoch') {
          setTrainingProgress((prev) => [...prev, {
            epoch: msg.epoch,
            totalEpochs: msg.totalEpochs,
            loss: msg.loss,
            accuracy: msg.accuracy,
            time: msg.time,
            batches: msg.batches,
            samples: msg.samples,
          }]);
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
            setStatus({ type: 'error', message: msg.error ?? 'Training failed' });
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
          g.nodes.delete(change.id);
          g.edges = g.edges.filter(
            (e) => e.source.nodeId !== change.id && e.target.nodeId !== change.id,
          );
          invalidateModel();
        }
      }
      setRFNodes((nds) => RF.applyNodeChanges(changes, nds));
    },
    [invalidateModel, getCurrentGraph],
  );

  const onEdgesChange = useCallback(
    (changes: RF.EdgeChange[]) => {
      const g = getCurrentGraph();
      for (const change of changes) {
        if (change.type === 'remove') {
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
    [invalidateModel, getCurrentGraph],
  );

  // Re-sync React Flow when navigating into/out of subgraphs
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { syncToRF(); }, [navStack]);

  return {
    graph: graphRef.current,
    currentGraph: getCurrentGraph(),
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
    runTrain,
    cancelTrain,
    status,
    modelTrained,
    modelStale,
    trainingProgress,
    savedBlocks,
    saveBlock,
    deleteBlock,
    addBlockFromTemplate,
  };
}
