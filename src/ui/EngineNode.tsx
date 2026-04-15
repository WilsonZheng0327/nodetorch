// Generic node component that renders ANY node type using data from the engine.
// Reads the NodeInstance (from Layer 1) and NodeDefinition (from Layer 4)
// to render ports, properties, and metadata.

import * as RF from '@xyflow/react';
import type { NodeInstance } from '../core/graph';
import type { DomainContext } from '../domain';
import { getNodePorts } from '../core/ports';
import { createContext, useContext, useRef, useEffect } from 'react';

// The domain context is provided at the top of the React tree
// so every node component can look up definitions.
export const DomainCtx = createContext<DomainContext | null>(null);

// Callback context for actions that nodes can trigger
export const GraphActionsCtx = createContext<{
  removeNode: (nodeId: string) => void;
} | null>(null);

export type EngineNodeData = {
  instance: NodeInstance;
};

const categoryColors: Record<string, string> = {
  Data: '#f59e0b',
  Layers: '#3b82f6',
  Activations: '#10b981',
  Loss: '#ef4444',
  Optimizers: '#8b5cf6',
};

function getCategoryColor(category: string[]): string {
  // Check from most specific to least specific
  for (let i = category.length - 1; i >= 0; i--) {
    if (categoryColors[category[i]]) return categoryColors[category[i]];
  }
  return '#6b7280';
}

export function EngineNode({ data, id }: RF.NodeProps<RF.Node<EngineNodeData>>) {
  const domain = useContext(DomainCtx);
  const actions = useContext(GraphActionsCtx);
  if (!domain) return null;

  const { instance } = data;
  const def = domain.nodeRegistry.get(instance.type);
  if (!def) {
    return <div className="layer-node">Unknown: {instance.type}</div>;
  }

  const ports = getNodePorts(instance, domain.nodeRegistry);
  const inputPorts = ports.filter((p) => p.direction === 'input');
  const outputPorts = ports.filter((p) => p.direction === 'output');
  const metadata = instance.lastResult?.metadata;
  const color = def.color ?? getCategoryColor(def.category);

  return (
    <div className="layer-node">
      {actions && (
        <button
          className="node-delete-btn"
          onClick={() => actions.removeNode(id)}
          title="Delete node"
        >
          &ndash;
        </button>
      )}
      <div className="layer-node-header" style={{ backgroundColor: color }}>
        {instance.type === 'subgraph.block' ? (instance.properties.blockName || def.displayName) : def.displayName}
      </div>
      <div className="layer-node-body">
        {/* Properties */}
        {Object.entries(instance.properties).map(([key, value]) => (
          <div key={key} className="layer-node-prop">
            <span className="prop-key">{key}</span>
            <span className="prop-value">{String(value)}</span>
          </div>
        ))}

        {/* Error message */}
        {metadata?.error && (
          <div className="layer-node-error">{metadata.error}</div>
        )}

        {/* Labeled shapes — e.g. "Output: [1, 64, 26, 26]", "Labels: [1]" */}
        {metadata?.shapes ? (
          metadata.shapes.map((s: { label: string; value: any[] | string }) => (
            <div key={s.label} className={`layer-node-shape ${metadata?.error ? 'layer-node-shape-error' : ''}`}>
              <span className="shape-label">{s.label}</span>
              <span>{Array.isArray(s.value) ? `[${s.value.join(', ')}]` : s.value}</span>
            </div>
          ))
        ) : metadata?.outputShape ? (
          <div className={`layer-node-shape ${metadata?.error ? 'layer-node-shape-error' : ''}`}>
            <span>{Array.isArray(metadata.outputShape) ? `[${metadata.outputShape.join(', ')}]` : metadata.outputShape}</span>
          </div>
        ) : !metadata?.finalLoss && !metadata?.prediction && !metadata?.actualLabel ? (
          <div className="layer-node-shape">
            <span>—</span>
          </div>
        ) : null}

        {/* Param count from metadata */}
        {metadata?.paramCount != null && (
          <div className="layer-node-params">
            {metadata.paramCount.toLocaleString()} params
          </div>
        )}

        {/* Training results for optimizer nodes */}
        {metadata?.finalLoss != null && (
          <div className="layer-node-shape">
            <span className="shape-label">Loss</span>
            <span>{metadata.finalLoss.toFixed(4)}</span>
          </div>
        )}
        {metadata?.finalAccuracy != null && (
          <div className="layer-node-shape layer-node-shape-success">
            <span className="shape-label">Accuracy</span>
            <span>{(metadata.finalAccuracy * 100).toFixed(1)}%</span>
          </div>
        )}

        {/* Inference: image preview + actual label on data nodes */}
        {metadata?.imagePixels && (
          <ImagePreview pixels={metadata.imagePixels} size={56} channels={metadata.imageChannels} />
        )}
        {metadata?.actualLabel != null && (
          <div className="layer-node-shape">
            <span className="shape-label">Label</span>
            <span>{metadata.actualLabel}</span>
          </div>
        )}

        {/* Inference: prediction on the final layer */}
        {metadata?.prediction && (
          <div className={`layer-node-shape ${
            metadata.prediction.predictedClass === metadata.prediction._actualLabel
              ? 'layer-node-shape-success' : ''
          }`}>
            <span className="shape-label">Predicted</span>
            <span>{metadata.prediction.predictedClass} ({(metadata.prediction.confidence * 100).toFixed(1)}%)</span>
          </div>
        )}

        {/* Forward pass tensor stats */}
        {metadata?.forwardResults?.out && (
          <div className="layer-node-stats">
            <span>mean {metadata.forwardResults.out.mean?.toFixed(3)}</span>
            <span>std {metadata.forwardResults.out.std?.toFixed(3)}</span>
          </div>
        )}

        {/* Input ports — inside body so they center on body, not header */}
        {inputPorts.map((port, i) => (
          <div key={port.id} className="port port-input" style={{ top: `${((i + 1) / (inputPorts.length + 1)) * 100}%` }}>
            <RF.Handle
              id={port.id}
              type="target"
              position={RF.Position.Left}
            />
            <span className="port-label port-label-left">{port.name}</span>
          </div>
        ))}

        {/* Output ports */}
        {outputPorts.map((port, i) => (
          <div key={port.id} className="port port-output" style={{ top: `${((i + 1) / (outputPorts.length + 1)) * 100}%` }}>
            <span className="port-label port-label-right">{port.name}</span>
            <RF.Handle
              id={port.id}
              type="source"
              position={RF.Position.Right}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Image preview (renders pixel data to a small canvas) ---

function ImagePreview({ pixels, size, channels }: { pixels: number[][] | number[][][]; size: number; channels?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pixels || pixels.length === 0) return;

    const h = pixels.length;
    const w = (pixels[0] as number[]).length ?? (pixels[0] as number[][]).length;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imageData = ctx.createImageData(w, h);
    const isRGB = channels && channels >= 3;

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (isRGB) {
          const px = (pixels as number[][][])[y][x];
          imageData.data[idx] = px[0];
          imageData.data[idx + 1] = px[1];
          imageData.data[idx + 2] = px[2];
        } else {
          const v = (pixels as number[][])[y][x];
          imageData.data[idx] = v;
          imageData.data[idx + 1] = v;
          imageData.data[idx + 2] = v;
        }
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [pixels, channels]);

  return (
    <div className="layer-node-image">
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size, imageRendering: 'pixelated' }}
      />
    </div>
  );
}
