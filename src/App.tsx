import * as RF from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './App.css';

import { useMemo, useEffect, useCallback, useState, useRef, type DragEvent } from 'react';
import { initDomain } from './domain';
import { useGraph } from './ui/useGraph';
import { EngineNode, DomainCtx, GraphActionsCtx, VizCtx, BackpropCtx } from './ui/EngineNode';
import { PropertyInspector } from './ui/inspector/PropertyInspector';
import { NodePalette } from './ui/NodePalette';
import { Toolbar } from './ui/Toolbar';
import { TrainingDashboard, type ModelLayerInfo } from './ui/dashboard/TrainingDashboard';
import { StepThroughPanel } from './ui/step-through/StepThroughPanel';
import { ShortcutsHelp } from './ui/ShortcutsHelp';
import { Breadcrumb } from './ui/Breadcrumb';
import { createNode as cn, addNode as an, createEdge as ce, addEdge as ae } from './core/graph';
import { Sun, Moon } from 'lucide-react';
import { TutorialPanel, tutorialEvent } from './ui/tutorial/TutorialPanel';

const categoryColors: Record<string, string> = {
  Data: '#f59e0b',
  Layers: '#3b82f6',
  Activations: '#10b981',
  Loss: '#ef4444',
  Optimizers: '#8b5cf6',
};

function getCategoryColor(category: string[]): string {
  for (let i = category.length - 1; i >= 0; i--) {
    if (categoryColors[category[i]]) return categoryColors[category[i]];
  }
  return '#6b7280';
}

// All node types use the same generic component.
// React Flow maps node.type to a component — we register every
// node type in the registry to the same EngineNode renderer.
function useNodeTypes(domain: ReturnType<typeof initDomain>) {
  return useMemo(() => {
    const types: Record<string, typeof EngineNode> = {};
    for (const def of domain.nodeRegistry.list()) {
      types[def.type] = EngineNode;
    }
    return types;
  }, [domain]);
}

export default function App() {
  const domain = useMemo(() => initDomain(), []);
  const nodeTypes = useNodeTypes(domain);
  const graph = useGraph(domain);

  const graphActions = useMemo(() => ({
    removeNode: graph.removeNode,
  }), [graph.removeNode]);

  const vizCtx = useMemo(() => ({
    pinnedVizNodes: graph.pinnedVizNodes,
    toggleVizPin: graph.toggleVizPin,
    liveSnapshots: graph.liveSnapshots,
  }), [graph.pinnedVizNodes, graph.toggleVizPin, graph.liveSnapshots]);


  // Model summary derived from graph nodes (after shape inference)
  const modelSummary = useMemo((): ModelLayerInfo[] => {
    const layers: ModelLayerInfo[] = [];
    for (const node of graph.graph.nodes.values()) {
      const def = domain.nodeRegistry.get(node.type);
      if (!def) continue;
      const meta = node.lastResult?.metadata;
      layers.push({
        name: def.displayName,
        type: node.type.split('.').pop() ?? node.type,
        paramCount: meta?.paramCount,
        outputShape: meta?.outputShape,
      });
    }
    return layers;
  }, [graph.rfNodes, domain]);

  const [selectedNodeIds, setSelectedNodeIds] = useState<string[]>([]);
  const [stepThroughOpen, setStepThroughOpen] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [dashboardOpen, setDashboardOpen] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('nodetorch-theme');
    return (saved === 'light') ? 'light' : 'dark';
  });
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('nodetorch-theme', theme);
  }, [theme]);

  // Only the Train button opens the dashboard automatically.
  const isTraining = graph.trainingActive;
  useEffect(() => {
    if (isTraining) setDashboardOpen(true);
  }, [isTraining]);
  const selectedNode = selectedNodeIds.length === 1
    ? graph.currentGraph.nodes.get(selectedNodeIds[0]) ?? null
    : null;

  // Track selected node IDs (supports multi-select)
  const onSelectionChange = useCallback(({ nodes }: { nodes: RF.Node[] }) => {
    setSelectedNodeIds(nodes.map((n) => n.id));
  }, []);

  // Explicit node click — ensures selection registers even if onSelectionChange is delayed
  const onNodeClick = useCallback((_event: React.MouseEvent, node: RF.Node) => {
    setSelectedNodeIds((prev) => {
      // If already the only selected node, keep it (avoid unnecessary re-render)
      if (prev.length === 1 && prev[0] === node.id) return prev;
      return [node.id];
    });
  }, []);

  // Right-click an edge to delete it
  const onEdgeContextMenu = useCallback(
    (event: React.MouseEvent, edge: RF.Edge) => {
      event.preventDefault();
      graph.onEdgesChange([{ id: edge.id, type: 'remove' }]);
      graph.runShape();
    },
    [graph],
  );

  // Double-click a subgraph node to enter it
  const onNodeDoubleClick = useCallback((_event: React.MouseEvent, node: RF.Node) => {
    const instance = graph.currentGraph.nodes.get(node.id);
    if (instance?.type === 'subgraph.block' && instance.subgraph) {
      graph.enterSubgraph(node.id);
    }
  }, [graph]);

  // Drag-and-drop from palette onto canvas
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<RF.ReactFlowInstance | null>(null);

  // Arrow keys / WASD for smooth panning
  useEffect(() => {
    const PAN_SPEED = 8; // pixels per frame
    const keyDirs: Record<string, { x: number; y: number }> = {
      ArrowUp: { x: 0, y: 1 }, ArrowDown: { x: 0, y: -1 },
      ArrowLeft: { x: 1, y: 0 }, ArrowRight: { x: -1, y: 0 },
      w: { x: 0, y: 1 }, s: { x: 0, y: -1 },
      a: { x: 1, y: 0 }, d: { x: -1, y: 0 },
    };

    const held = new Set<string>();
    let frameId: number | null = null;

    function tick() {
      if (!reactFlowInstance || held.size === 0) {
        frameId = null;
        return;
      }
      let dx = 0, dy = 0;
      for (const key of held) {
        const dir = keyDirs[key];
        if (dir) { dx += dir.x; dy += dir.y; }
      }
      if (dx !== 0 || dy !== 0) {
        const { x, y, zoom } = reactFlowInstance.getViewport();
        reactFlowInstance.setViewport({
          x: x + dx * PAN_SPEED,
          y: y + dy * PAN_SPEED,
          zoom,
        });
      }
      frameId = requestAnimationFrame(tick);
    }

    function onDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      // Shortcuts help
      if (e.key === '?' || (e.key === '/' && e.shiftKey)) {
        e.preventDefault();
        setShortcutsOpen((v) => !v);
        return;
      }

      // F toggles the training dashboard
      if (e.key === 'f' || e.key === 'F') {
        e.preventDefault();
        setDashboardOpen((v) => !v);
        return;
      }

      // Undo/Redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        graph.undo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'Z' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        graph.redo();
        return;
      }

      // Copy/Paste
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        e.preventDefault();
        graph.copySelected();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        e.preventDefault();
        graph.paste();
        return;
      }

      if (keyDirs[e.key]) {
        e.preventDefault();
        held.add(e.key);
        if (!frameId) frameId = requestAnimationFrame(tick);
      }
    }

    function onUp(e: KeyboardEvent) {
      held.delete(e.key);
    }

    window.addEventListener('keydown', onDown);
    window.addEventListener('keyup', onUp);
    return () => {
      window.removeEventListener('keydown', onDown);
      window.removeEventListener('keyup', onUp);
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [reactFlowInstance]);

  const onDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();
      if (!reactFlowInstance) return;

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Check if it's a saved block
      const blockFilename = event.dataTransfer.getData('application/nodetorch-block');
      if (blockFilename) {
        graph.addBlockFromTemplate(blockFilename, position);
        return;
      }

      // Otherwise it's a regular node type
      const nodeType = event.dataTransfer.getData('application/nodetorch-type');
      if (nodeType) {
        graph.addNode(nodeType, position);
      }
    },
    [reactFlowInstance, graph],
  );

  // Helper: build an identity ResBlock (same channels in/out, stride 1)
  // Input → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → Add(skip) → ReLU → Output
  function buildResBlock(block: { subgraph: any }, channels: number) {
    const inner = block.subgraph;
    inner.nodes.get('output')!.position = { x: 1100, y: 100 };

    const conv1 = cn('conv1', 'ml.layers.conv2d', { x: 150, y: 100 }, { outChannels: channels, kernelSize: 3, stride: 1, padding: 1 });
    const bn1 = cn('bn1', 'ml.layers.batchnorm2d', { x: 300, y: 100 }, {});
    const relu1 = cn('relu1', 'ml.activations.relu', { x: 440, y: 100 }, {});
    const conv2 = cn('conv2', 'ml.layers.conv2d', { x: 580, y: 100 }, { outChannels: channels, kernelSize: 3, stride: 1, padding: 1 });
    const bn2 = cn('bn2', 'ml.layers.batchnorm2d', { x: 730, y: 100 }, {});
    const add = cn('add', 'ml.structural.add', { x: 870, y: 100 }, {});
    const relu2 = cn('relu2', 'ml.activations.relu', { x: 990, y: 100 }, {});

    [conv1, bn1, relu1, conv2, bn2, add, relu2].forEach((n) => an(inner, n));

    ae(inner, ce('r1', 'input', 'in', 'conv1', 'in'));
    ae(inner, ce('r2', 'conv1', 'out', 'bn1', 'in'));
    ae(inner, ce('r3', 'bn1', 'out', 'relu1', 'in'));
    ae(inner, ce('r4', 'relu1', 'out', 'conv2', 'in'));
    ae(inner, ce('r5', 'conv2', 'out', 'bn2', 'in'));
    ae(inner, ce('r6', 'bn2', 'out', 'add', 'a'));
    ae(inner, ce('r7', 'input', 'in', 'add', 'b'));       // identity skip
    ae(inner, ce('r8', 'add', 'out', 'relu2', 'in'));
    ae(inner, ce('r9', 'relu2', 'out', 'output', 'out'));
  }

  // Helper: build a downsampling ResBlock (channels double, spatial halved via stride 2)
  // Main path: Conv(3×3, stride 2) → BN → ReLU → Conv(3×3) → BN
  // Skip path: Conv(1×1, stride 2) → BN  (projection shortcut)
  // Add → ReLU → Output
  function buildResBlockDown(block: { subgraph: any }, outChannels: number) {
    const inner = block.subgraph;
    inner.nodes.get('output')!.position = { x: 1300, y: 100 };

    // Main path
    const conv1 = cn('conv1', 'ml.layers.conv2d', { x: 150, y: 60 }, { outChannels: outChannels, kernelSize: 3, stride: 2, padding: 1 });
    const bn1 = cn('bn1', 'ml.layers.batchnorm2d', { x: 300, y: 60 }, {});
    const relu1 = cn('relu1', 'ml.activations.relu', { x: 440, y: 60 }, {});
    const conv2 = cn('conv2', 'ml.layers.conv2d', { x: 580, y: 60 }, { outChannels: outChannels, kernelSize: 3, stride: 1, padding: 1 });
    const bn2 = cn('bn2', 'ml.layers.batchnorm2d', { x: 730, y: 60 }, {});

    // Projection shortcut (1×1 conv to match channels + stride 2 to match spatial)
    const projConv = cn('proj_conv', 'ml.layers.conv2d', { x: 300, y: 250 }, { outChannels: outChannels, kernelSize: 1, stride: 2, padding: 0 });
    const projBn = cn('proj_bn', 'ml.layers.batchnorm2d', { x: 500, y: 250 }, {});

    // Merge
    const add = cn('add', 'ml.structural.add', { x: 900, y: 100 }, {});
    const relu2 = cn('relu2', 'ml.activations.relu', { x: 1060, y: 100 }, {});

    [conv1, bn1, relu1, conv2, bn2, projConv, projBn, add, relu2].forEach((n) => an(inner, n));

    // Main path
    ae(inner, ce('r1', 'input', 'in', 'conv1', 'in'));
    ae(inner, ce('r2', 'conv1', 'out', 'bn1', 'in'));
    ae(inner, ce('r3', 'bn1', 'out', 'relu1', 'in'));
    ae(inner, ce('r4', 'relu1', 'out', 'conv2', 'in'));
    ae(inner, ce('r5', 'conv2', 'out', 'bn2', 'in'));
    ae(inner, ce('r6', 'bn2', 'out', 'add', 'a'));
    // Projection skip
    ae(inner, ce('r7', 'input', 'in', 'proj_conv', 'in'));
    ae(inner, ce('r8', 'proj_conv', 'out', 'proj_bn', 'in'));
    ae(inner, ce('r9', 'proj_bn', 'out', 'add', 'b'));
    // Merge → output
    ae(inner, ce('r10', 'add', 'out', 'relu2', 'in'));
    ae(inner, ce('r11', 'relu2', 'out', 'output', 'out'));
  }

  // Build a full ResNet-34 for CIFAR-100
  // Architecture: [3, 4, 6, 3] BasicBlocks at [64, 128, 256, 512] channels
  // CIFAR-adapted stem: 3×3 conv stride 1 (no 7×7 stride 2, no maxpool)
  // Default demo graph disabled — users start with an empty canvas and the tutorial.
  // To restore, uncomment the block below.
  // Load a preset on startup
  const demoLoaded = useRef(false);
  useEffect(() => {
    if (demoLoaded.current) return;
    demoLoaded.current = true;
    fetch('http://localhost:8000/presets/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: 'diffusion-mnist.json' }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') {
          graph.loadGraph(JSON.stringify(data.data));
          setTimeout(() => reactFlowInstance?.fitView({ padding: 0.2 }), 200);
        }
      })
      .catch(() => { /* backend not running yet */ });
  }, []);

  // Fit view when navigating into/out of subgraphs
  useEffect(() => {
    // Use requestAnimationFrame so React Flow has measured the new nodes,
    // then fitView with duration:0 for instant snap (no visible flash)
    requestAnimationFrame(() => {
      reactFlowInstance?.fitView({ padding: 0.3, duration: 0 });
    });
  }, [graph.navStack, reactFlowInstance]);

  return (
    <DomainCtx.Provider value={domain}>
    <GraphActionsCtx.Provider value={graphActions}>
    <VizCtx.Provider value={vizCtx}>
    <BackpropCtx.Provider value={graph.backpropAnim}>
      <div ref={reactFlowWrapper} style={{ width: '100vw', height: '100vh' }}>
        <RF.ReactFlow
          nodes={graph.rfNodes}
          edges={graph.rfEdges}
          onNodesChange={graph.onNodesChange}
          onEdgesChange={graph.onEdgesChange}
          onConnect={graph.connect}
          isValidConnection={graph.isValidConnection}
          onEdgeClick={(_e, _edge) => {/* no-op: right-click only */}}
          onEdgeContextMenu={onEdgeContextMenu}
          onNodeClick={onNodeClick}
          onNodeDoubleClick={onNodeDoubleClick}
          onSelectionChange={onSelectionChange}
          onInit={setReactFlowInstance}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodeTypes={nodeTypes}
          edgesFocusable={false}
          selectionKeyCode="Control"
          multiSelectionKeyCode="Shift"
          fitView
          defaultEdgeOptions={{ animated: true }}
        >
          <RF.Background />
          <RF.Panel position="bottom-right" className="bottom-panel">
            <RF.MiniMap
              className="minimap-themed"
              maskColor="rgba(17, 17, 27, 0.7)"
              nodeColor={(node) => {
                const inst = graph.graph.nodes.get(node.id);
                if (!inst) return '#45475a';
                const def = domain.nodeRegistry.get(inst.type);
                if (!def) return '#45475a';
                return def.color ?? getCategoryColor(def.category);
              }}
            />
            <RF.Controls
              orientation="horizontal"
              showInteractive={false}
              className="controls-themed"
              position="bottom-right"
            />
          </RF.Panel>
          <RF.Panel position="bottom-left">
            <button
              className="theme-toggle"
              onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </RF.Panel>
        </RF.ReactFlow>
        {graph.connectionError && (
          <div className="connection-error-toast">{graph.connectionError}</div>
        )}
        <Toolbar onSave={graph.saveGraph} onLoad={(json: string) => { graph.loadGraph(json); setTimeout(() => reactFlowInstance?.fitView({ padding: 0.2 }), 200); }} onClear={graph.clearGraph} onOrganize={graph.organizeGraph} onShowAllViz={graph.showAllViz} onHideAllViz={graph.hideAllViz} onStepThrough={() => { setStepThroughOpen(true); tutorialEvent('step-through-opened'); }} onSimulateBackprop={graph.simulateBackprop} onSaveModel={graph.saveModel} onLoadModel={graph.loadModel} onExportPython={graph.exportPython} onInfer={graph.runInfer} onTest={graph.runTest} onTrain={graph.runTrain} onCancel={graph.cancelTrain} status={graph.status} modelTrained={graph.modelTrained} modelStale={graph.modelStale} />
        <input
          ref={graph.weightsInputRef}
          type="file"
          accept=".pt"
          style={{ display: 'none' }}
          onChange={graph.handleWeightsFile}
        />
        <Breadcrumb navStack={graph.navStack} onNavigate={graph.navigateTo} />
        <TutorialPanel />
        <NodePalette
          savedBlocks={graph.savedBlocks}
          onDeleteBlock={graph.deleteBlock}
        />
        <PropertyInspector
          node={selectedNode}
          selectedCount={selectedNodeIds.length}
          onPropertyChange={graph.updateProperty}
          onSaveBlock={graph.saveBlock}
          graphJson={graph.saveGraph()}
        />
        <TrainingDashboard
          progress={graph.trainingProgress}
          isTraining={isTraining}
          batchProgress={graph.batchProgress}
          selectedEpoch={graph.selectedEpoch}
          onSelectEpoch={graph.setSelectedEpoch}
          totalSnapshotEpochs={graph.snapshotHistory.length}
          modelSummary={modelSummary}
          testResult={graph.testResult}
          open={dashboardOpen}
          onOpenChange={setDashboardOpen}
        />
        <StepThroughPanel
          open={stepThroughOpen}
          graphJson={graph.saveGraph()}
          onClose={() => setStepThroughOpen(false)}
        />
        <ShortcutsHelp open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      </div>
    </BackpropCtx.Provider>
    </VizCtx.Provider>
    </GraphActionsCtx.Provider>
    </DomainCtx.Provider>
  );
}
