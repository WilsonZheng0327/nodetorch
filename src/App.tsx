import * as RF from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import './App.css';

import { useMemo, useEffect, useCallback, useState, useRef, type DragEvent } from 'react';
import { initDomain } from './domain';
import { useGraph } from './ui/useGraph';
import { EngineNode, DomainCtx, GraphActionsCtx } from './ui/EngineNode';
import { PropertyInspector } from './ui/PropertyInspector';
import { NodePalette } from './ui/NodePalette';
import { Toolbar } from './ui/Toolbar';
import { TrainingDashboard } from './ui/TrainingDashboard';
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


  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
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

  // Build a CIFAR-100 demo with a custom conv block
  const demoBuilt = useRef(false);
  useEffect(() => {
    if (demoBuilt.current) return;
    demoBuilt.current = true;
    async function buildDemo() {
      // Outer graph: CIFAR-100 → ConvBlock → Flatten → Linear → Loss → Adam
      await graph.addNode('data.cifar100', { x: 0, y: 200 });
      await graph.addNode('subgraph.block', { x: 300, y: 180 });
      await graph.addNode('ml.layers.flatten', { x: 560, y: 180 });
      await graph.addNode('ml.layers.linear', { x: 760, y: 180 });
      await graph.addNode('ml.loss.cross_entropy', { x: 1000, y: 230 });
      await graph.addNode('ml.optimizers.adam', { x: 1260, y: 280 });

      const nodes = Array.from(graph.graph.nodes.values());
      const cifar = nodes.find((n) => n.type === 'data.cifar100');
      const block = nodes.find((n) => n.type === 'subgraph.block');
      const flatten = nodes.find((n) => n.type === 'ml.layers.flatten');
      const linear = nodes.find((n) => n.type === 'ml.layers.linear');
      const loss = nodes.find((n) => n.type === 'ml.loss.cross_entropy');
      const adam = nodes.find((n) => n.type === 'ml.optimizers.adam');

      if (!cifar || !block || !flatten || !linear || !loss || !adam) return;

      // Configure nodes
      await graph.updateProperty(block.id, 'blockName', 'Conv Block');
      await graph.updateProperty(linear.id, 'outFeatures', 100);

      // Build the inner graph: Input → Conv(32,3,pad=1) → BN → ReLU → MaxPool → Output
      const inner = block.subgraph!;
      const innerOutput = inner.nodes.get('output')!;

      // Add inner nodes
      const conv = cn('conv', 'ml.layers.conv2d', { x: 150, y: 80 }, { outChannels: 32, kernelSize: 3, stride: 1, padding: 1 });
      const bn = cn('bn', 'ml.layers.batchnorm2d', { x: 350, y: 80 }, {});
      const relu = cn('relu', 'ml.activations.relu', { x: 500, y: 80 }, {});
      const pool = cn('pool', 'ml.layers.maxpool2d', { x: 650, y: 80 }, { kernelSize: 2, stride: 2, padding: 0 });
      an(inner, conv);
      an(inner, bn);
      an(inner, relu);
      an(inner, pool);

      // Move output node further right
      innerOutput.position = { x: 850, y: 80 };

      // Wire inner graph
      ae(inner, ce('ie1', 'input', 'in', 'conv', 'in'));
      ae(inner, ce('ie2', 'conv', 'out', 'bn', 'in'));
      ae(inner, ce('ie3', 'bn', 'out', 'relu', 'in'));
      ae(inner, ce('ie4', 'relu', 'out', 'pool', 'in'));
      ae(inner, ce('ie5', 'pool', 'out', 'output', 'out'));

      // Wire outer graph
      await graph.connect({ source: cifar.id, sourceHandle: 'out', target: block.id, targetHandle: 'in' });
      await graph.connect({ source: block.id, sourceHandle: 'out', target: flatten.id, targetHandle: 'in' });
      await graph.connect({ source: flatten.id, sourceHandle: 'out', target: linear.id, targetHandle: 'in' });
      await graph.connect({ source: linear.id, sourceHandle: 'out', target: loss.id, targetHandle: 'predictions' });
      await graph.connect({ source: cifar.id, sourceHandle: 'labels', target: loss.id, targetHandle: 'labels' });
      await graph.connect({ source: loss.id, sourceHandle: 'out', target: adam.id, targetHandle: 'loss' });
    }
    buildDemo().then(() => {
      setTimeout(() => reactFlowInstance?.fitView({ padding: 0.2 }), 100);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fit view when navigating into/out of subgraphs
  useEffect(() => {
    setTimeout(() => reactFlowInstance?.fitView({ padding: 0.3 }), 50);
  }, [graph.navStack, reactFlowInstance]);

  return (
    <DomainCtx.Provider value={domain}>
    <GraphActionsCtx.Provider value={graphActions}>
      <div ref={reactFlowWrapper} style={{ width: '100vw', height: '100vh' }}>
        <RF.ReactFlow
          nodes={graph.rfNodes}
          edges={graph.rfEdges}
          onNodesChange={graph.onNodesChange}
          onEdgesChange={graph.onEdgesChange}
          onConnect={graph.connect}
          isValidConnection={graph.isValidConnection}
          onEdgeContextMenu={onEdgeContextMenu}
          onNodeDoubleClick={onNodeDoubleClick}
          onSelectionChange={onSelectionChange}
          onInit={setReactFlowInstance}
          onDragOver={onDragOver}
          onDrop={onDrop}
          nodeTypes={nodeTypes}
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
        <Toolbar onSave={graph.saveGraph} onLoad={graph.loadGraph} onClear={graph.clearGraph} onRun={graph.runForward} onInfer={graph.runInfer} onTrain={graph.runTrain} onCancel={graph.cancelTrain} status={graph.status} modelTrained={graph.modelTrained} modelStale={graph.modelStale} />
        <Breadcrumb navStack={graph.navStack} onNavigate={graph.navigateTo} />
        <NodePalette
          savedBlocks={graph.savedBlocks}
          onDeleteBlock={graph.deleteBlock}
        />
        <PropertyInspector node={selectedNode} onPropertyChange={graph.updateProperty} onSaveBlock={graph.saveBlock} />
        <TrainingDashboard progress={graph.trainingProgress} isTraining={graph.status.type === 'running'} />
      </div>
    </GraphActionsCtx.Provider>
    </DomainCtx.Provider>
  );
}
