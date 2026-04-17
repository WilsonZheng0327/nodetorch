// LayerDetail modal — detailed visualizations for a specific node.
// Opened from PropertyInspector. Fetches data from /layer-detail endpoint.
// Renders weight matrix heatmap, feature maps, attention map, hidden state.

import { useState, useEffect, useRef } from 'react';
import './LayerDetail.css';

interface Props {
  nodeId: string;
  nodeType: string;
  graphJson: string;  // serialized graph to send to backend
  onClose: () => void;
}

interface DetailData {
  nodeType: string;
  weightMatrix?: { data: number[][]; rows: number; cols: number; actualRows: number; actualCols: number; min: number; max: number };
  convKernels?: { kernels: number[][][]; count: number; totalFilters: number; height: number; width: number; inChannels: number };
  featureMaps?: { maps: number[][][]; channels: number; height: number; width: number };
  attentionMap?: { data: number[][]; rows: number; cols: number };
  hiddenState?: { data: number[][]; rows: number; cols: number; label: string };
  cellState?: { data: number[][]; rows: number; cols: number; label: string };
  confusionMatrix?: { data: number[][]; size: number };
}

export function LayerDetail({ nodeId, nodeType, graphJson, onClose }: Props) {
  const [detail, setDetail] = useState<DetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch('http://localhost:8000/layer-detail', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), nodeId }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setDetail(data.detail);
        else setError(data.error ?? 'Failed to load detail');
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  }, [nodeId, graphJson]);

  const typeName = nodeType.split('.').pop() ?? nodeType;

  return (
    <div className="layer-detail-overlay" onClick={onClose}>
      <div className="layer-detail-modal" onClick={(e) => e.stopPropagation()}>
        <div className="layer-detail-header">
          <span className="layer-detail-title">{typeName} — Detail</span>
          <button className="layer-detail-close" onClick={onClose}>&times;</button>
        </div>
        <div className="layer-detail-body">
          {loading && <div className="layer-detail-loading">Loading...</div>}
          {error && <div className="layer-detail-error">{error}</div>}
          {detail && (
            <>
              {detail.weightMatrix && (
                <DetailSection title={`Weight Matrix (${detail.weightMatrix.actualRows} x ${detail.weightMatrix.actualCols})`}>
                  <Heatmap
                    data={detail.weightMatrix.data}
                    rows={detail.weightMatrix.rows}
                    cols={detail.weightMatrix.cols}
                    min={detail.weightMatrix.min}
                    max={detail.weightMatrix.max}
                  />
                  <div className="heatmap-legend">
                    <span>{detail.weightMatrix.min.toFixed(4)}</span>
                    <div className="heatmap-legend-bar" />
                    <span>{detail.weightMatrix.max.toFixed(4)}</span>
                  </div>
                  {(detail.weightMatrix.rows < detail.weightMatrix.actualRows ||
                    detail.weightMatrix.cols < detail.weightMatrix.actualCols) && (
                    <div className="heatmap-note">
                      Downsampled from {detail.weightMatrix.actualRows}x{detail.weightMatrix.actualCols} for display
                    </div>
                  )}
                </DetailSection>
              )}

              {detail.convKernels && (
                <DetailSection title={`Conv Kernels (${detail.convKernels.count}/${detail.convKernels.totalFilters} filters, ${detail.convKernels.height}x${detail.convKernels.width}, ${detail.convKernels.inChannels}ch avg)`}>
                  <div className="feature-maps-grid">
                    {detail.convKernels.kernels.map((kernel, i) => (
                      <FeatureMapCanvas key={i} pixels={kernel} label={`f${i}`} />
                    ))}
                  </div>
                </DetailSection>
              )}

              {detail.featureMaps && (
                <DetailSection title={`Feature Maps (${detail.featureMaps.channels} channels)`}>
                  <div className="feature-maps-grid">
                    {detail.featureMaps.maps.map((fm, i) => (
                      <FeatureMapCanvas key={i} pixels={fm} label={`ch ${i}`} />
                    ))}
                  </div>
                </DetailSection>
              )}

              {detail.attentionMap && (
                <DetailSection title="Attention Scores">
                  <Heatmap
                    data={detail.attentionMap.data}
                    rows={detail.attentionMap.rows}
                    cols={detail.attentionMap.cols}
                    min={0}
                    max={1}
                    colorScheme="warm"
                  />
                </DetailSection>
              )}

              {detail.hiddenState && (
                <DetailSection title={detail.hiddenState.label}>
                  <Heatmap
                    data={detail.hiddenState.data}
                    rows={detail.hiddenState.rows}
                    cols={detail.hiddenState.cols}
                    colorScheme="diverging"
                  />
                </DetailSection>
              )}

              {detail.cellState && (
                <DetailSection title={detail.cellState.label}>
                  <Heatmap
                    data={detail.cellState.data}
                    rows={detail.cellState.rows}
                    cols={detail.cellState.cols}
                    colorScheme="diverging"
                  />
                </DetailSection>
              )}

              {detail.confusionMatrix && (
                <DetailSection title={`Confusion Matrix (${detail.confusionMatrix.size} classes)`}>
                  <ConfusionMatrixView data={detail.confusionMatrix.data} size={detail.confusionMatrix.size} />
                </DetailSection>
              )}

              {!detail.weightMatrix && !detail.featureMaps && !detail.attentionMap && !detail.hiddenState && !detail.confusionMatrix && (
                <div className="layer-detail-loading">No detailed visualization available for this node type</div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function DetailSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="detail-section">
      <div className="detail-section-title">{title}</div>
      {children}
    </div>
  );
}

// --- Heatmap canvas ---

function Heatmap({ data, rows, cols, min, max, colorScheme }: {
  data: number[][];
  rows: number;
  cols: number;
  min?: number;
  max?: number;
  colorScheme?: 'cool' | 'warm' | 'diverging';
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scheme = colorScheme ?? 'cool';

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const cellSize = Math.max(3, Math.min(8, Math.floor(400 / Math.max(rows, cols))));
    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const vmin = min ?? Math.min(...data.flat());
    const vmax = max ?? Math.max(...data.flat());
    const range = vmax - vmin || 1;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = (data[r]?.[c] ?? 0);
        const t = (v - vmin) / range; // 0..1
        ctx.fillStyle = heatColor(t, scheme);
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
  }, [data, rows, cols, min, max, scheme]);

  return <canvas ref={canvasRef} className="heatmap-canvas" />;
}

function heatColor(t: number, scheme: string): string {
  t = Math.max(0, Math.min(1, t));
  if (scheme === 'warm') {
    // Black → Red → Yellow → White
    const r = Math.round(Math.min(255, t * 2 * 255));
    const g = Math.round(Math.max(0, (t - 0.5) * 2 * 255));
    const b = Math.round(Math.max(0, (t - 0.75) * 4 * 255));
    return `rgb(${r},${g},${b})`;
  }
  if (scheme === 'diverging') {
    // Blue → White → Red (centered at 0.5)
    if (t < 0.5) {
      const s = t * 2;
      return `rgb(${Math.round(s * 255)},${Math.round(s * 255)},255)`;
    } else {
      const s = (t - 0.5) * 2;
      return `rgb(255,${Math.round((1 - s) * 255)},${Math.round((1 - s) * 255)})`;
    }
  }
  // cool: dark blue → cyan → green → yellow
  const r = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
  const g = Math.round(t < 0.5 ? t * 2 * 200 : 200 + (t - 0.5) * 2 * 55);
  const b = Math.round(t < 0.5 ? 100 + t * 2 * 155 : 255 - (t - 0.5) * 2 * 255);
  return `rgb(${r},${g},${b})`;
}

// --- Feature map small canvas ---

function FeatureMapCanvas({ pixels, label }: { pixels: number[][]; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0 || !pixels[0]) return;

    const h = pixels.length;
    const w = pixels[0].length;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const imageData = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const v = pixels[y][x];
        imageData.data[idx] = v;
        imageData.data[idx + 1] = v;
        imageData.data[idx + 2] = v;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }, [pixels]);

  return (
    <div className="feature-map-item">
      <canvas ref={canvasRef} className="feature-map-canvas" />
      <span className="feature-map-label">{label}</span>
    </div>
  );
}

// --- Confusion matrix grid ---

function ConfusionMatrixView({ data, size }: { data: number[][]; size: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const cellSize = Math.max(16, Math.min(28, Math.floor(400 / size)));
    const labelSpace = 30;
    canvas.width = size * cellSize + labelSpace;
    canvas.height = size * cellSize + labelSpace;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const maxVal = Math.max(...data.flat(), 1);

    // Column headers (predicted)
    ctx.fillStyle = '#6c7086';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let c = 0; c < size; c++) {
      ctx.fillText(String(c), labelSpace + c * cellSize + cellSize / 2, 10);
    }

    // Row labels (actual) + cells
    for (let r = 0; r < size; r++) {
      ctx.textAlign = 'right';
      ctx.fillStyle = '#6c7086';
      ctx.fillText(String(r), labelSpace - 4, labelSpace + r * cellSize + cellSize / 2 + 3);

      for (let c = 0; c < size; c++) {
        const v = data[r][c];
        const t = v / maxVal;
        const x = labelSpace + c * cellSize;
        const y = labelSpace + r * cellSize;

        // Diagonal = correct (green), off-diagonal = errors (red)
        if (r === c) {
          ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + t * 0.8})`;
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${0.05 + t * 0.7})`;
        }
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // Value text
        if (v > 0) {
          ctx.fillStyle = t > 0.5 ? '#fff' : '#a6adc8';
          ctx.font = '9px JetBrains Mono, monospace';
          ctx.textAlign = 'center';
          ctx.fillText(String(v), x + cellSize / 2, y + cellSize / 2 + 3);
        }
      }
    }

    // Axis labels
    ctx.fillStyle = '#6c7086';
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Predicted', labelSpace + (size * cellSize) / 2, size * cellSize + labelSpace + 12);
  }, [data, size]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <canvas ref={canvasRef} style={{ borderRadius: 4 }} />
      <div style={{ fontSize: 9, color: '#6c7086', marginTop: 4 }}>
        Rows = Actual, Columns = Predicted
      </div>
    </div>
  );
}
