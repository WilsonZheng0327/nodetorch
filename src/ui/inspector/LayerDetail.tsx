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
  misclassifications?: {
    actual: number;
    predicted: number;
    confidence: number;
    imagePixels: number[][] | number[][][];
    imageChannels: number;
  }[];
}

interface LandscapeData {
  grid: number[][];
  alphaRange: number;
  gridSize: number;
  centerLoss: number;
  minLoss: number;
  maxLoss: number;
  usedTrainedWeights: boolean;
}

interface ActMaxData {
  dreams: { pixels: number[] | number[][]; channels: number; filterIndex: number }[];
  totalFilters: number;
  iterations: number;
  usingTrainedWeights: boolean;
}

export function LayerDetail({ nodeId, nodeType, graphJson, onClose }: Props) {
  const [detail, setDetail] = useState<DetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [actMax, setActMax] = useState<ActMaxData | null>(null);
  const [actMaxLoading, setActMaxLoading] = useState(false);
  const [misclassFilter, setMisclassFilter] = useState<{ actual: number; predicted: number } | null>(null);
  const [landscape, setLandscape] = useState<LandscapeData | null>(null);
  const [landscapeLoading, setLandscapeLoading] = useState(false);

  const runLandscape = () => {
    setLandscapeLoading(true);
    fetch('http://localhost:8000/loss-landscape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), gridSize: 11, alphaRange: 1.0 }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setLandscape(data.result);
        else setError(data.error ?? 'Failed to compute loss landscape');
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLandscapeLoading(false));
  };

  const runActivationMax = () => {
    setActMaxLoading(true);
    fetch('http://localhost:8000/activation-max', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), nodeId, numFilters: 8, iterations: 25 }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setActMax(data.result);
        else setError(data.error ?? 'Failed to run activation max');
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setActMaxLoading(false));
  };

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

              {/* Activation maximization — only available for conv-like layers */}
              {detail.convKernels && (
                <DetailSection title="Activation Maximization (what each filter 'wants to see')">
                  {!actMax && !actMaxLoading && (
                    <button className="layer-detail-action-btn" onClick={runActivationMax}>
                      Generate Dream Images
                    </button>
                  )}
                  {actMaxLoading && (
                    <div className="layer-detail-loading">
                      Running gradient ascent (25 steps × 8 filters)...
                    </div>
                  )}
                  {actMax && (
                    <>
                      <div className="heatmap-note">
                        {actMax.usingTrainedWeights
                          ? 'Dreams from trained filters — each image maximizes one filter\'s activation'
                          : 'Dreams from random (untrained) filters — will look like noise'}
                      </div>
                      <div className="feature-maps-grid">
                        {actMax.dreams.map((d, i) => (
                          <DreamCanvas key={i} pixels={d.pixels} channels={d.channels} label={`f${d.filterIndex}`} />
                        ))}
                      </div>
                    </>
                  )}
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
                  <ConfusionMatrixView
                    data={detail.confusionMatrix.data}
                    size={detail.confusionMatrix.size}
                    onCellClick={(actual, predicted) => {
                      if (actual === predicted) {
                        setMisclassFilter(null);
                      } else {
                        setMisclassFilter({ actual, predicted });
                      }
                    }}
                    highlightCell={misclassFilter}
                  />
                </DetailSection>
              )}

              {detail.misclassifications && detail.misclassifications.length > 0 && (
                <DetailSection title={
                  misclassFilter
                    ? `Misclassifications — actual ${misclassFilter.actual} predicted as ${misclassFilter.predicted}`
                    : `Misclassified Samples (${detail.misclassifications.length})`
                }>
                  {misclassFilter && (
                    <button
                      className="layer-detail-action-btn"
                      onClick={() => setMisclassFilter(null)}
                      style={{ marginBottom: 8 }}
                    >
                      ← Show all
                    </button>
                  )}
                  <div className="misclass-grid">
                    {detail.misclassifications
                      .filter((m) =>
                        !misclassFilter ||
                        (m.actual === misclassFilter.actual && m.predicted === misclassFilter.predicted)
                      )
                      .slice(0, 32)
                      .map((m, i) => (
                        <MisclassCard key={i} sample={m} />
                      ))}
                  </div>
                  {misclassFilter && detail.misclassifications.filter((m) =>
                    m.actual === misclassFilter.actual && m.predicted === misclassFilter.predicted
                  ).length === 0 && (
                    <div className="layer-detail-loading">No samples stored for this pair</div>
                  )}
                </DetailSection>
              )}

              {nodeType.startsWith('ml.loss.') && (
                <DetailSection title="Loss Landscape (2D slice around current weights)">
                  {!landscape && !landscapeLoading && (
                    <>
                      <div className="heatmap-note">
                        Projects loss onto two random directions in weight space.
                        Bowl shape = converged. Chaotic = under-trained.
                      </div>
                      <button className="layer-detail-action-btn" onClick={runLandscape}>
                        Compute Loss Landscape (11×11 grid)
                      </button>
                    </>
                  )}
                  {landscapeLoading && (
                    <div className="layer-detail-loading">
                      Computing 121 loss evaluations...
                    </div>
                  )}
                  {landscape && (
                    <LossLandscapeView data={landscape} />
                  )}
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

// --- Loss landscape heatmap ---

function LossLandscapeView({ data }: { data: LandscapeData }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const cellSize = 20;
    const padding = 24;
    const w = data.gridSize * cellSize + padding * 2;
    const h = data.gridSize * cellSize + padding * 2;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const range = data.maxLoss - data.minLoss || 1;

    // Draw heatmap
    for (let r = 0; r < data.gridSize; r++) {
      for (let c = 0; c < data.gridSize; c++) {
        const v = data.grid[r][c];
        const t = (v - data.minLoss) / range;
        // Warm-to-cool: low loss (good) = cool blue, high loss = warm red
        // Invert: low t = cool, high t = warm
        const r255 = Math.round(Math.min(255, t * 255));
        const g255 = Math.round(Math.min(255, Math.abs(t - 0.5) * 2 * 255));
        const b255 = Math.round(Math.min(255, (1 - t) * 255));
        ctx.fillStyle = `rgb(${r255}, ${g255}, ${b255})`;
        ctx.fillRect(padding + c * cellSize, padding + r * cellSize, cellSize, cellSize);
      }
    }

    // Center marker (current weights)
    const cx = padding + (data.gridSize / 2) * cellSize;
    const cy = padding + (data.gridSize / 2) * cellSize;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.stroke();

    // Axes labels
    ctx.fillStyle = '#cdd6f4';
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`α = -${data.alphaRange}`, padding + cellSize / 2, padding - 4);
    ctx.fillText(`α = +${data.alphaRange}`, padding + (data.gridSize - 0.5) * cellSize, padding - 4);
    ctx.textAlign = 'right';
    ctx.fillText(`β = -${data.alphaRange}`, padding - 4, padding + cellSize);
    ctx.fillText(`β = +${data.alphaRange}`, padding - 4, padding + (data.gridSize - 0.5) * cellSize + 4);
  }, [data]);

  return (
    <div>
      <canvas ref={canvasRef} style={{ borderRadius: 4, border: '1px solid #45475a', imageRendering: 'pixelated' }} />
      <div className="heatmap-legend">
        <span>min: {data.minLoss.toFixed(4)} (blue)</span>
        <span style={{ flex: 1 }}></span>
        <span>center: {data.centerLoss.toFixed(4)}</span>
        <span style={{ flex: 1 }}></span>
        <span>max: {data.maxLoss.toFixed(4)} (red)</span>
      </div>
      <div className="heatmap-note">
        {data.usedTrainedWeights
          ? 'Computed around trained weights. Smooth valley = well-converged. Sharp ravines = unstable minimum.'
          : 'Computed around random init (loss is high everywhere). Train first for a meaningful landscape.'}
      </div>
    </div>
  );
}

// --- Dream canvas (for activation maximization — supports grayscale and RGB) ---

function DreamCanvas({ pixels, channels, label }: { pixels: number[] | number[][]; channels: number; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;

    const isRGB = channels >= 3;
    const h = (pixels as number[][]).length;
    const w = isRGB ? (pixels as any)[0].length : (pixels as number[][])[0].length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (isRGB) {
          const px = (pixels as any)[y][x];
          img.data[idx] = px[0];
          img.data[idx + 1] = px[1];
          img.data[idx + 2] = px[2];
        } else {
          const v = (pixels as number[][])[y][x];
          img.data[idx] = v;
          img.data[idx + 1] = v;
          img.data[idx + 2] = v;
        }
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels, channels]);

  return (
    <div className="feature-map-item">
      <canvas ref={canvasRef} className="feature-map-canvas" style={{ width: 64, height: 64 }} />
      <span className="feature-map-label">{label}</span>
    </div>
  );
}

// --- Confusion matrix grid ---

function ConfusionMatrixView({ data, size, onCellClick, highlightCell }: {
  data: number[][];
  size: number;
  onCellClick?: (actual: number, predicted: number) => void;
  highlightCell?: { actual: number; predicted: number } | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cellSizeRef = useRef(16);
  const labelSpaceRef = useRef(30);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const cellSize = Math.max(16, Math.min(28, Math.floor(400 / size)));
    const labelSpace = 30;
    cellSizeRef.current = cellSize;
    labelSpaceRef.current = labelSpace;
    canvas.width = size * cellSize + labelSpace;
    canvas.height = size * cellSize + labelSpace;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const maxVal = Math.max(...data.flat(), 1);

    ctx.fillStyle = '#6c7086';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let c = 0; c < size; c++) {
      ctx.fillText(String(c), labelSpace + c * cellSize + cellSize / 2, 10);
    }

    for (let r = 0; r < size; r++) {
      ctx.textAlign = 'right';
      ctx.fillStyle = '#6c7086';
      ctx.fillText(String(r), labelSpace - 4, labelSpace + r * cellSize + cellSize / 2 + 3);

      for (let c = 0; c < size; c++) {
        const v = data[r][c];
        const t = v / maxVal;
        const x = labelSpace + c * cellSize;
        const y = labelSpace + r * cellSize;

        if (r === c) {
          ctx.fillStyle = `rgba(16, 185, 129, ${0.1 + t * 0.8})`;
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${0.05 + t * 0.7})`;
        }
        ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

        // Highlighted cell gets a border
        if (highlightCell && highlightCell.actual === r && highlightCell.predicted === c) {
          ctx.strokeStyle = '#89b4fa';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, cellSize - 1, cellSize - 1);
          ctx.lineWidth = 1;
        }

        if (v > 0) {
          ctx.fillStyle = t > 0.5 ? '#fff' : '#a6adc8';
          ctx.font = '9px JetBrains Mono, monospace';
          ctx.textAlign = 'center';
          ctx.fillText(String(v), x + cellSize / 2, y + cellSize / 2 + 3);
        }
      }
    }

    ctx.fillStyle = '#6c7086';
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Predicted', labelSpace + (size * cellSize) / 2, size * cellSize + labelSpace + 12);
  }, [data, size, highlightCell]);

  function handleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!onCellClick) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    const c = Math.floor((x - labelSpaceRef.current) / cellSizeRef.current);
    const r = Math.floor((y - labelSpaceRef.current) / cellSizeRef.current);
    if (r >= 0 && r < size && c >= 0 && c < size) {
      onCellClick(r, c);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <canvas
        ref={canvasRef}
        style={{ borderRadius: 4, cursor: onCellClick ? 'pointer' : 'default' }}
        onClick={handleClick}
      />
      <div style={{ fontSize: 9, color: '#6c7086', marginTop: 4 }}>
        Rows = Actual, Columns = Predicted{onCellClick ? ' — click a cell to filter misclassifications' : ''}
      </div>
    </div>
  );
}

// --- Misclassification card (shown in gallery) ---

function MisclassCard({ sample }: { sample: {
  actual: number; predicted: number; confidence: number;
  imagePixels: number[][] | number[][][]; imageChannels: number;
} }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const pixels = sample.imagePixels;
    if (pixels.length === 0) return;
    const h = pixels.length;
    const isRGB = sample.imageChannels >= 3;
    const w = isRGB ? (pixels as number[][][])[0].length : (pixels as number[][])[0].length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (isRGB) {
          const px = (pixels as number[][][])[y][x];
          data.data[idx] = px[0];
          data.data[idx + 1] = px[1];
          data.data[idx + 2] = px[2];
        } else {
          const v = (pixels as number[][])[y][x];
          data.data[idx] = v;
          data.data[idx + 1] = v;
          data.data[idx + 2] = v;
        }
        data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [sample]);
  return (
    <div className="misclass-card">
      <canvas ref={canvasRef} className="misclass-canvas" />
      <div className="misclass-labels">
        <div className="misclass-actual">actual {sample.actual}</div>
        <div className="misclass-predicted">pred {sample.predicted} ({(sample.confidence * 100).toFixed(0)}%)</div>
      </div>
    </div>
  );
}
