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
import { Breadcrumb } from './ui/Breadcrumb';
import { createNode as cn, addNode as an, createEdge as ce, addEdge as ae } from './core/graph';

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

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [stepThroughOpen, setStepThroughOpen] = useState(false);
  const selectedNode = selectedNodeId ? graph.currentGraph.nodes.get(selectedNodeId) ?? null : null;

  // Track which node is selected
  const onSelectionChange = useCallback(({ nodes }: { nodes: RF.Node[] }) => {
    setSelectedNodeId(nodes.length === 1 ? nodes[0].id : null);
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

  // Helper: build a ResBlock subgraph inside a block node
  // Input → Conv → BN → ReLU → Conv → BN → Add(skip) → ReLU → Output
  function buildResBlock(block: { subgraph: any }, channels: number) {
    const inner = block.subgraph;
    const innerOutput = inner.nodes.get('output')!;
    innerOutput.position = { x: 1100, y: 100 };

    // Main path
    const conv1 = cn('conv1', 'ml.layers.conv2d', { x: 150, y: 100 }, { outChannels: channels, kernelSize: 3, stride: 1, padding: 1 });
    const bn1 = cn('bn1', 'ml.layers.batchnorm2d', { x: 300, y: 100 }, {});
    const relu1 = cn('relu1', 'ml.activations.relu', { x: 440, y: 100 }, {});
    const conv2 = cn('conv2', 'ml.layers.conv2d', { x: 580, y: 100 }, { outChannels: channels, kernelSize: 3, stride: 1, padding: 1 });
    const bn2 = cn('bn2', 'ml.layers.batchnorm2d', { x: 730, y: 100 }, {});

    // Skip connection + merge
    const add = cn('add', 'ml.structural.add', { x: 870, y: 100 }, {});
    const relu2 = cn('relu2', 'ml.activations.relu', { x: 990, y: 100 }, {});

    an(inner, conv1); an(inner, bn1); an(inner, relu1);
    an(inner, conv2); an(inner, bn2);
    an(inner, add); an(inner, relu2);

    // Main path wiring
    ae(inner, ce('r1', 'input', 'in', 'conv1', 'in'));
    ae(inner, ce('r2', 'conv1', 'out', 'bn1', 'in'));
    ae(inner, ce('r3', 'bn1', 'out', 'relu1', 'in'));
    ae(inner, ce('r4', 'relu1', 'out', 'conv2', 'in'));
    ae(inner, ce('r5', 'conv2', 'out', 'bn2', 'in'));
    ae(inner, ce('r6', 'bn2', 'out', 'add', 'a'));

    // Skip connection: input → add.b
    ae(inner, ce('r7', 'input', 'in', 'add', 'b'));

    // Merge → output
    ae(inner, ce('r8', 'add', 'out', 'relu2', 'in'));
    ae(inner, ce('r9', 'relu2', 'out', 'output', 'out'));
  }

  // Build a small ResNet for CIFAR-100
  const demoBuilt = useRef(false);
  useEffect(() => {
    if (demoBuilt.current) return;
    demoBuilt.current = true;
    async function buildDemo() {
      // Stem: CIFAR-100 → Conv(16) → BN → ReLU
      await graph.addNode('data.cifar100', { x: 0, y: 200 });
      await graph.addNode('ml.layers.conv2d', { x: 240, y: 200 });
      await graph.addNode('ml.layers.batchnorm2d', { x: 440, y: 200 });
      await graph.addNode('ml.activations.relu', { x: 600, y: 200 });

      // ResBlock × 2
      await graph.addNode('subgraph.block', { x: 780, y: 200 });
      await graph.addNode('subgraph.block', { x: 1000, y: 200 });

      // Head: AdaptiveAvgPool → Flatten → Linear(100) → Loss → Adam
      await graph.addNode('ml.layers.adaptive_avgpool2d', { x: 1220, y: 200 });
      await graph.addNode('ml.layers.flatten', { x: 1420, y: 200 });
      await graph.addNode('ml.layers.linear', { x: 1600, y: 200 });
      await graph.addNode('ml.loss.cross_entropy', { x: 1820, y: 250 });
      await graph.addNode('ml.optimizers.adam', { x: 2060, y: 300 });

      const nodes = Array.from(graph.graph.nodes.values());
      const cifar = nodes.find((n) => n.type === 'data.cifar100')!;
      const stemConv = nodes.find((n) => n.type === 'ml.layers.conv2d')!;
      const stemBn = nodes.find((n) => n.type === 'ml.layers.batchnorm2d')!;
      const stemRelu = nodes.find((n) => n.type === 'ml.activations.relu')!;
      const resBlock1 = nodes.filter((n) => n.type === 'subgraph.block')[0]!;
      const resBlock2 = nodes.filter((n) => n.type === 'subgraph.block')[1]!;
      const avgpool = nodes.find((n) => n.type === 'ml.layers.adaptive_avgpool2d')!;
      const flatten = nodes.find((n) => n.type === 'ml.layers.flatten')!;
      const linear = nodes.find((n) => n.type === 'ml.layers.linear')!;
      const loss = nodes.find((n) => n.type === 'ml.loss.cross_entropy')!;
      const adam = nodes.find((n) => n.type === 'ml.optimizers.adam')!;

      // Configure stem conv: 3→16, 3x3, pad=1
      await graph.updateProperty(stemConv.id, 'outChannels', 16);
      await graph.updateProperty(stemConv.id, 'kernelSize', 3);
      await graph.updateProperty(stemConv.id, 'padding', 1);

      // Configure ResBlocks
      await graph.updateProperty(resBlock1.id, 'blockName', 'ResBlock 1');
      await graph.updateProperty(resBlock2.id, 'blockName', 'ResBlock 2');
      buildResBlock(resBlock1 as any, 16);
      buildResBlock(resBlock2 as any, 16);

      // Configure head
      await graph.updateProperty(linear.id, 'outFeatures', 100);

      // Wire the outer graph
      await graph.connect({ source: cifar.id, sourceHandle: 'out', target: stemConv.id, targetHandle: 'in' });
      await graph.connect({ source: stemConv.id, sourceHandle: 'out', target: stemBn.id, targetHandle: 'in' });
      await graph.connect({ source: stemBn.id, sourceHandle: 'out', target: stemRelu.id, targetHandle: 'in' });
      await graph.connect({ source: stemRelu.id, sourceHandle: 'out', target: resBlock1.id, targetHandle: 'in' });
      await graph.connect({ source: resBlock1.id, sourceHandle: 'out', target: resBlock2.id, targetHandle: 'in' });
      await graph.connect({ source: resBlock2.id, sourceHandle: 'out', target: avgpool.id, targetHandle: 'in' });
      await graph.connect({ source: avgpool.id, sourceHandle: 'out', target: flatten.id, targetHandle: 'in' });
      await graph.connect({ source: flatten.id, sourceHandle: 'out', target: linear.id, targetHandle: 'in' });
      await graph.connect({ source: linear.id, sourceHandle: 'out', target: loss.id, targetHandle: 'predictions' });
      await graph.connect({ source: cifar.id, sourceHandle: 'labels', target: loss.id, targetHandle: 'labels' });
      await graph.connect({ source: loss.id, sourceHandle: 'out', target: adam.id, targetHandle: 'loss' });

      // Re-run shape inference now that inner subgraphs are fully built
      await graph.runShape();
    }
    buildDemo().then(() => {
      setTimeout(() => reactFlowInstance?.fitView({ padding: 0.2 }), 100);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
          onNodeDoubleClick={onNodeDoubleClick}
          onSelectionChange={onSelectionChange}
          onInit={setReactFlowInstance}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodeTypes={nodeTypes}
          edgesFocusable={false}
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
        </RF.ReactFlow>
        <Toolbar onSave={graph.saveGraph} onLoad={graph.loadGraph} onClear={graph.clearGraph} onOrganize={graph.organizeGraph} onShowAllViz={graph.showAllViz} onHideAllViz={graph.hideAllViz} onStepThrough={() => setStepThroughOpen(true)} onSimulateBackprop={graph.simulateBackprop} onRun={graph.runForward} onInfer={graph.runInfer} onTrain={graph.runTrain} onCancel={graph.cancelTrain} status={graph.status} modelTrained={graph.modelTrained} modelStale={graph.modelStale} />
        <Breadcrumb navStack={graph.navStack} onNavigate={graph.navigateTo} />
        <NodePalette
          savedBlocks={graph.savedBlocks}
          onDeleteBlock={graph.deleteBlock}
        />
        <PropertyInspector node={selectedNode} onPropertyChange={graph.updateProperty} onSaveBlock={graph.saveBlock} graphJson={graph.saveGraph()} />
        <TrainingDashboard
          progress={graph.trainingProgress}
          isTraining={graph.status.type === 'running'}
          batchProgress={graph.batchProgress}
          selectedEpoch={graph.selectedEpoch}
          onSelectEpoch={graph.setSelectedEpoch}
          totalSnapshotEpochs={graph.snapshotHistory.length}
          modelSummary={modelSummary}
        />
        <StepThroughPanel
          open={stepThroughOpen}
          graphJson={graph.saveGraph()}
          onClose={() => setStepThroughOpen(false)}
        />
      </div>
    </BackpropCtx.Provider>
    </VizCtx.Provider>
    </GraphActionsCtx.Provider>
    </DomainCtx.Provider>
  );
}
