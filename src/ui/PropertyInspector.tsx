// Property inspector panel — shows editable properties for the selected node.
// Auto-generates widgets from PropertyDefinition.type (number → input, boolean → toggle, etc.)

import { useContext, useRef, useEffect, useState } from 'react';
import type { NodeInstance } from '../core/graph';
import type { PropertyDefinition } from '../core/nodedef';
import { DomainCtx } from './EngineNode';
import { DatasetDetail } from './DatasetDetail';

interface Props {
  node: NodeInstance | null;
  onPropertyChange: (nodeId: string, key: string, value: any) => void;
  onSaveBlock?: (nodeId: string) => void;
}

export function PropertyInspector({ node, onPropertyChange, onSaveBlock }: Props) {
  const domain = useContext(DomainCtx);
  const [datasetDetailType, setDatasetDetailType] = useState<string | null>(null);

  if (!node || !domain) {
    return (
      <>
        <div className="inspector">
          <div className="inspector-empty">Select a node to edit properties</div>
        </div>
        {datasetDetailType && (
          <DatasetDetail datasetType={datasetDetailType} onClose={() => setDatasetDetailType(null)} />
        )}
      </>
    );
  }

  const def = domain.nodeRegistry.get(node.type);
  if (!def) return null;

  const properties = def.getProperties();
  const metadata = node.lastResult?.metadata;
  const isDataNode = metadata?.datasetType;

  return (
    <>
    {datasetDetailType && (
      <DatasetDetail datasetType={datasetDetailType} onClose={() => setDatasetDetailType(null)} />
    )}
    <div className="inspector">
      <div className="inspector-header">{def.displayName}</div>
      <div className="inspector-desc">{def.description}</div>

      {properties.length > 0 && (
        <div className="inspector-section">
          <div className="inspector-section-title">Properties</div>
          {properties.map((prop) => {
            if (prop.visible && !prop.visible(node.properties)) return null;

            return (
              <PropertyWidget
                key={prop.id}
                definition={prop}
                value={node.properties[prop.id]}
                onChange={(value) => onPropertyChange(node.id, prop.id, value)}
              />
            );
          })}
        </div>
      )}

      {metadata && (
        <div className="inspector-section">
          <div className="inspector-section-title">Info</div>
          {metadata.outputShape && (
            <div className="inspector-info">
              <span>Output Shape</span>
              <span>[{metadata.outputShape.join(', ')}]</span>
            </div>
          )}
          {metadata.paramCount != null && (
            <div className="inspector-info">
              <span>Parameters</span>
              <span>{metadata.paramCount.toLocaleString()}</span>
            </div>
          )}
          {metadata.paramBreakdown && (
            <div className="inspector-info-breakdown">
              {metadata.paramBreakdown}
            </div>
          )}
          {metadata.error && (
            <div className="inspector-info inspector-info-error">
              <span>{metadata.error}</span>
            </div>
          )}
          {metadata.prediction && (
            <>
              <div className="inspector-info">
                <span>Predicted</span>
                <span>{metadata.prediction.predictedClass} ({(metadata.prediction.confidence * 100).toFixed(1)}%)</span>
              </div>
              {metadata.prediction.probabilities && (
                <ProbabilityBars probs={metadata.prediction.probabilities} predicted={metadata.prediction.predictedClass} />
              )}
            </>
          )}
        </div>
      )}
      {metadata?.weights && (
        <div className="inspector-section">
          <div className="inspector-section-title">Weights</div>
          <div className="inspector-info">
            <span>Mean</span><span>{metadata.weights.mean?.toFixed(4)}</span>
          </div>
          <div className="inspector-info">
            <span>Std</span><span>{metadata.weights.std?.toFixed(4)}</span>
          </div>
          <div className="inspector-info">
            <span>Range</span><span>{metadata.weights.min?.toFixed(4)} to {metadata.weights.max?.toFixed(4)}</span>
          </div>
          {metadata.weights.histBins && (
            <Histogram bins={metadata.weights.histBins} counts={metadata.weights.histCounts} color="#89b4fa" label="Weight distribution" />
          )}
        </div>
      )}
      {metadata?.activations && (
        <div className="inspector-section">
          <div className="inspector-section-title">Activations</div>
          <div className="inspector-info">
            <span>Mean</span><span>{metadata.activations.mean?.toFixed(4)}</span>
          </div>
          <div className="inspector-info">
            <span>Std</span><span>{metadata.activations.std?.toFixed(4)}</span>
          </div>
          <div className="inspector-info">
            <span>Sparsity</span><span>{metadata.activations.sparsity != null ? `${(metadata.activations.sparsity * 100).toFixed(1)}%` : '—'}</span>
          </div>
          {metadata.activations.histBins && (
            <Histogram bins={metadata.activations.histBins} counts={metadata.activations.histCounts} color="#10b981" label="Activation distribution" />
          )}
        </div>
      )}
      {metadata?.imagePixels && (
        <div className="inspector-section">
          <div className="inspector-section-title">Input Image</div>
          <InspectorImage pixels={metadata.imagePixels} channels={metadata.imageChannels} />
          {metadata.actualLabel != null && (
            <div className="inspector-info" style={{ marginTop: 6 }}>
              <span>Actual Label</span>
              <span>{metadata.actualLabel}</span>
            </div>
          )}
        </div>
      )}
      {isDataNode && (
        <div className="inspector-section">
          <button
            className="inspector-dataset-btn"
            onClick={() => setDatasetDetailType(metadata.datasetType)}
          >
            View Dataset Details
          </button>
        </div>
      )}
      {node.type === 'subgraph.block' && node.subgraph && onSaveBlock && (
        <div className="inspector-section">
          <button
            className="inspector-dataset-btn"
            onClick={() => onSaveBlock(node.id)}
          >
            Save Block to Library
          </button>
        </div>
      )}
    </div>
    </>
  );
}

// --- Property Widgets ---

interface WidgetProps {
  definition: PropertyDefinition;
  value: any;
  onChange: (value: any) => void;
}

function PropertyWidget({ definition, value, onChange }: WidgetProps) {
  const { type } = definition;
  const currentValue = value ?? definition.defaultValue;

  // Clamp a number value to min/max if defined
  function clamp(v: number): number {
    if (type.kind !== 'number') return v;
    if (type.min != null && v < type.min) return type.min;
    if (type.max != null && v > type.max) return type.max;
    return v;
  }

  return (
    <div className="inspector-prop">
      <label className="inspector-prop-label">{definition.name}</label>
      <div className="inspector-prop-detail">{describeType(definition)}</div>
      {type.kind === 'number' && (
        <div className="inspector-number-row">
          <button
            className="inspector-step-btn"
            onClick={() => onChange(clamp(currentValue - (type.step ?? 1)))}
          >
            -
          </button>
          <input
            type="number"
            className="inspector-input"
            value={currentValue}
            min={type.min}
            max={type.max}
            step={type.step ?? (type.integer ? 1 : undefined)}
            onChange={(e) => {
              const v = type.integer ? parseInt(e.target.value) : parseFloat(e.target.value);
              if (!isNaN(v)) onChange(clamp(v));
            }}
          />
          <button
            className="inspector-step-btn"
            onClick={() => onChange(clamp(currentValue + (type.step ?? 1)))}
          >
            +
          </button>
        </div>
      )}
      {type.kind === 'boolean' && (
        <button
          className={`inspector-toggle ${currentValue ? 'inspector-toggle-on' : ''}`}
          onClick={() => onChange(!currentValue)}
        >
          <span className="inspector-toggle-thumb" />
          <span className="inspector-toggle-label">{currentValue ? 'On' : 'Off'}</span>
        </button>
      )}
      {type.kind === 'select' && (
        <select
          className="inspector-select"
          value={currentValue}
          onChange={(e) => onChange(e.target.value)}
        >
          {type.options.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      )}
      {type.kind === 'string' && (
        <input
          type="text"
          className="inspector-input inspector-input-text"
          value={currentValue}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </div>
  );
}

// Generate a human-readable description of the property constraints
function describeType(def: PropertyDefinition): string {
  const { type } = def;
  const parts: string[] = [];

  if (type.kind === 'number') {
    if (type.integer) parts.push('integer');
    if (type.min != null && type.max != null) parts.push(`${type.min}–${type.max}`);
    else if (type.min != null) parts.push(`min ${type.min}`);
    else if (type.max != null) parts.push(`max ${type.max}`);
    if (type.step != null) parts.push(`step ${type.step}`);
  }

  if (def.affects === 'ports') parts.push('affects ports');
  else if (def.affects === 'both') parts.push('affects ports + shape');
  else if (def.affects === 'execution') parts.push('affects shape');

  return parts.join(' · ');
}

// --- Image preview for inspector (larger) ---

function InspectorImage({ pixels, channels }: { pixels: number[][] | number[][][]; channels?: number }) {
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

    const isRGB = channels && channels >= 3;
    const imageData = ctx.createImageData(w, h);
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
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        style={{ width: 112, height: 112, imageRendering: 'pixelated', borderRadius: 4, border: '1px solid #45475a' }}
      />
    </div>
  );
}

// --- Probability bar chart for predictions ---

function ProbabilityBars({ probs, predicted }: { probs: number[]; predicted: number }) {
  const maxProb = Math.max(...probs);

  return (
    <div className="inspector-probs">
      {probs.map((p, i) => (
        <div key={i} className="inspector-prob-row">
          <span className={`inspector-prob-label ${i === predicted ? 'inspector-prob-predicted' : ''}`}>
            {i}
          </span>
          <div className="inspector-prob-bar-bg">
            <div
              className={`inspector-prob-bar ${i === predicted ? 'inspector-prob-bar-active' : ''}`}
              style={{ width: `${(p / maxProb) * 100}%` }}
            />
          </div>
          <span className="inspector-prob-value">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

// --- Histogram ---

function Histogram({ bins, counts, color, label }: { bins: number[]; counts: number[]; color: string; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || bins.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const maxCount = Math.max(...counts);
    const barWidth = w / counts.length;

    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = color;
    ctx.globalAlpha = 0.6;
    for (let i = 0; i < counts.length; i++) {
      const barH = (counts[i] / maxCount) * (h - 2);
      ctx.fillRect(i * barWidth, h - barH, barWidth - 1, barH);
    }
    ctx.globalAlpha = 1;
  }, [bins, counts, color]);

  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ fontSize: 10, color: '#6c7086', marginBottom: 2 }}>{label}</div>
      <canvas ref={canvasRef} style={{ width: '100%', height: 50, borderRadius: 4, background: '#313244' }} />
    </div>
  );
}
