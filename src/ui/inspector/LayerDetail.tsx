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
  confusionMatrix?: { data: number[][]; size: number; classNames?: string[] };
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
  const [latentGrid, setLatentGrid] = useState<{ grid: (number[][] | number[][][] | null)[][]; gridSize: number; latentRange: number; imageH: number; imageW: number; channels: number } | null>(null);
  const [latentGridLoading, setLatentGridLoading] = useState(false);

  const runLatentGrid = () => {
    setLatentGridLoading(true);
    fetch('http://localhost:8000/latent-grid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), gridSize: 10, latentRange: 3.0 }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setLatentGrid(data.result);
        else setError(data.error ?? 'Failed to generate latent grid');
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLatentGridLoading(false));
  };

  const runLandscape = () => {
    setLandscapeLoading(true);
    fetch('http://localhost:8000/loss-landscape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), gridSize: 21, alphaRange: 10.0 }),
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
                    classNames={detail.confusionMatrix.classNames}
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

              {nodeType === 'ml.structural.reparameterize' && (
                <DetailSection title="Latent Space Grid">
                  {!latentGrid && !latentGridLoading && (
                    <>
                      <div className="heatmap-note">
                        Sweeps across two latent dimensions and decodes each point.
                        Shows how the model organizes concepts in the learned latent space.
                      </div>
                      <button className="layer-detail-action-btn" onClick={runLatentGrid}>
                        Generate Latent Grid (10×10)
                      </button>
                    </>
                  )}
                  {latentGridLoading && (
                    <div className="layer-detail-loading">Decoding 100 latent points...</div>
                  )}
                  {latentGrid && (
                    <LatentGridView data={latentGrid} />
                  )}
                </DetailSection>
              )}

              {!detail.weightMatrix && !detail.featureMaps && !detail.attentionMap && !detail.hiddenState && !detail.confusionMatrix && nodeType !== 'ml.structural.reparameterize' && (
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

    const padLeft = 64;         // room for β-axis labels
    const padTop = 40;          // room for α-axis labels + title
    const padRight = 24;
    const padBottom = 56;       // room for x-axis title

    // Fit to container width
    const containerW = canvas.parentElement?.clientWidth ?? 700;
    const cellSize = Math.max(12, Math.floor((containerW - padLeft - padRight) / data.gridSize));
    const plotW = data.gridSize * cellSize;
    const plotH = data.gridSize * cellSize;
    const w = plotW + padLeft + padRight;
    const h = plotH + padTop + padBottom;

    // Render at DPR for crispness
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const range = data.maxLoss - data.minLoss || 1;

    // Heatmap cells (blue = low loss, red = high loss)
    for (let r = 0; r < data.gridSize; r++) {
      for (let c = 0; c < data.gridSize; c++) {
        const v = data.grid[r][c];
        const t = (v - data.minLoss) / range;
        const r255 = Math.round(Math.min(255, t * 255));
        const g255 = Math.round(Math.min(255, Math.abs(t - 0.5) * 2 * 255));
        const b255 = Math.round(Math.min(255, (1 - t) * 255));
        ctx.fillStyle = `rgb(${r255}, ${g255}, ${b255})`;
        ctx.fillRect(padLeft + c * cellSize, padTop + r * cellSize, cellSize, cellSize);
      }
    }

    // Grid lines for readability
    ctx.strokeStyle = 'rgba(0,0,0,0.15)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= data.gridSize; i++) {
      ctx.beginPath();
      ctx.moveTo(padLeft + i * cellSize + 0.5, padTop);
      ctx.lineTo(padLeft + i * cellSize + 0.5, padTop + plotH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(padLeft, padTop + i * cellSize + 0.5);
      ctx.lineTo(padLeft + plotW, padTop + i * cellSize + 0.5);
      ctx.stroke();
    }

    // Center marker (current weights)
    const cx = padLeft + (data.gridSize / 2) * cellSize;
    const cy = padTop + (data.gridSize / 2) * cellSize;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 7, 0, Math.PI * 2);
    ctx.stroke();
    // Inner dot
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fill();
    // "current" label next to marker
    ctx.fillStyle = '#fff';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('current weights', cx + 10, cy + 4);

    // --- Axis ticks and labels ---
    ctx.fillStyle = '#cdd6f4';
    ctx.font = '12px JetBrains Mono, monospace';

    // α axis (horizontal) — bottom
    ctx.textAlign = 'center';
    ctx.fillText(`-${data.alphaRange.toFixed(1)}`, padLeft, padTop + plotH + 16);
    ctx.fillText('0', padLeft + plotW / 2, padTop + plotH + 16);
    ctx.fillText(`+${data.alphaRange.toFixed(1)}`, padLeft + plotW, padTop + plotH + 16);

    // β axis (vertical) — left side
    ctx.textAlign = 'right';
    ctx.fillText(`+${data.alphaRange.toFixed(1)}`, padLeft - 8, padTop + 5);
    ctx.fillText('0', padLeft - 8, padTop + plotH / 2 + 4);
    ctx.fillText(`-${data.alphaRange.toFixed(1)}`, padLeft - 8, padTop + plotH);

    // Axis titles
    ctx.fillStyle = '#a6adc8';
    ctx.font = '13px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('α  (random direction 1)', padLeft + plotW / 2, padTop + plotH + 38);
    ctx.save();
    ctx.translate(16, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('β  (random direction 2)', 0, 0);
    ctx.restore();
  }, [data]);

  return (
    <div>
      <div className="confusion-scroll">
        <canvas ref={canvasRef} style={{ borderRadius: 4, border: '1px solid #45475a', display: 'block' }} />
      </div>
      <div className="heatmap-legend">
        <span>low loss</span>
        <div className="landscape-color-bar" />
        <span>high loss</span>
      </div>
      <div className="heatmap-legend" style={{ marginTop: 4 }}>
        <span>min: {data.minLoss.toFixed(4)}</span>
        <span style={{ flex: 1 }}></span>
        <span>center: {data.centerLoss.toFixed(4)}</span>
        <span style={{ flex: 1 }}></span>
        <span>max: {data.maxLoss.toFixed(4)}</span>
      </div>
      <div className="heatmap-note">
        <strong>How to read this:</strong> we pick two random perturbation directions in
        weight space (α, β) and evaluate the loss at a grid of offsets from the current
        weights. The white circle at the center is your model's current position. Colors
        show the loss at each perturbed weight configuration — blue regions are lower loss,
        red are higher.{' '}
        {data.usedTrainedWeights
          ? 'A smooth blue valley around the center means the trained solution is stable; sharp ravines mean a brittle minimum that small weight changes would escape.'
          : 'These weights are random init, so loss is high everywhere — train the model first for a meaningful landscape.'}
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

function ConfusionMatrixView({ data, size, classNames, onCellClick, highlightCell }: {
  data: number[][];
  size: number;
  classNames?: string[];
  onCellClick?: (actual: number, predicted: number) => void;
  highlightCell?: { actual: number; predicted: number } | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cellSizeRef = useRef(16);
  const labelSpaceRef = useRef(30);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const hasNames = classNames && classNames.length >= size && size <= 20;
    const cellSize = Math.max(28, Math.min(40, Math.floor(600 / size)));
    const labelSpace = hasNames ? 80 : 40;
    cellSizeRef.current = cellSize;
    labelSpaceRef.current = labelSpace;
    canvas.width = size * cellSize + labelSpace;
    canvas.height = size * cellSize + labelSpace + 18;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const maxVal = Math.max(...data.flat(), 1);

    // Column headers
    ctx.fillStyle = '#a6adc8';
    ctx.font = hasNames ? '10px Inter, system-ui, sans-serif' : '12px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    for (let c = 0; c < size; c++) {
      const label = hasNames ? classNames![c] : String(c);
      ctx.save();
      ctx.translate(labelSpace + c * cellSize + cellSize / 2, labelSpace - 6);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = 'left';
      ctx.fillText(label, 0, 0);
      ctx.restore();
    }

    for (let r = 0; r < size; r++) {
      // Row headers
      ctx.textAlign = 'right';
      ctx.fillStyle = '#a6adc8';
      ctx.font = hasNames ? '10px Inter, system-ui, sans-serif' : '12px JetBrains Mono, monospace';
      const rowLabel = hasNames ? classNames![r] : String(r);
      ctx.fillText(rowLabel, labelSpace - 6, labelSpace + r * cellSize + cellSize / 2 + 4);

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
          ctx.fillStyle = t > 0.5 ? '#fff' : '#cdd6f4';
          ctx.font = '11px JetBrains Mono, monospace';
          ctx.textAlign = 'center';
          ctx.fillText(String(v), x + cellSize / 2, y + cellSize / 2 + 4);
        }
      }
    }

    ctx.fillStyle = '#a6adc8';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Predicted', labelSpace + (size * cellSize) / 2, size * cellSize + labelSpace + 14);
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
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
      {/* Scrollable wrapper: confusion matrix starts top-left and scrolls right/down
          when it exceeds the modal width. */}
      <div className="confusion-scroll">
        <canvas
          ref={canvasRef}
          style={{ borderRadius: 4, cursor: onCellClick ? 'pointer' : 'default', display: 'block' }}
          onClick={handleClick}
        />
      </div>
      <div style={{ fontSize: 12, color: '#a6adc8', marginTop: 6 }}>
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


// --- Latent Grid View (VAE) ---

function LatentGridView({ data }: { data: { grid: (number[][] | number[][][] | null)[][]; gridSize: number; latentRange: number; imageH: number; imageW: number; channels: number } }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.imageH === 0) return;

    const cellSize = Math.max(20, Math.min(40, Math.floor(600 / data.gridSize)));
    const padLeft = 40;
    const padTop = 24;
    const padRight = 8;
    const padBottom = 40;
    const totalW = data.gridSize * cellSize + padLeft + padRight;
    const totalH = data.gridSize * cellSize + padTop + padBottom;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = totalW * dpr;
    canvas.height = totalH * dpr;
    canvas.style.width = `${totalW}px`;
    canvas.style.height = `${totalH}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Black background
    ctx.fillStyle = '#11111b';
    ctx.fillRect(0, 0, totalW, totalH);

    // Render each cell
    for (let r = 0; r < data.gridSize; r++) {
      for (let c = 0; c < data.gridSize; c++) {
        const pixels = data.grid[r]?.[c];
        if (!pixels) continue;

        const x = padLeft + c * cellSize;
        const y = padTop + r * cellSize;
        const imgData = ctx.createImageData(data.imageW, data.imageH);

        for (let py = 0; py < data.imageH; py++) {
          for (let px = 0; px < data.imageW; px++) {
            const idx = (py * data.imageW + px) * 4;
            if (data.channels === 1) {
              const v = (pixels as number[][])[py]?.[px] ?? 0;
              imgData.data[idx] = v;
              imgData.data[idx + 1] = v;
              imgData.data[idx + 2] = v;
            } else {
              const p = (pixels as number[][][])[py]?.[px] ?? [0, 0, 0];
              imgData.data[idx] = p[0];
              imgData.data[idx + 1] = p[1];
              imgData.data[idx + 2] = p[2];
            }
            imgData.data[idx + 3] = 255;
          }
        }

        // Draw the small image scaled into the cell
        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = data.imageW;
        tmpCanvas.height = data.imageH;
        tmpCanvas.getContext('2d')!.putImageData(imgData, 0, 0);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tmpCanvas, x, y, cellSize - 1, cellSize - 1);
      }
    }

    // Axis labels
    ctx.fillStyle = '#a6adc8';
    ctx.font = '11px JetBrains Mono, monospace';

    // X axis (dim 0)
    ctx.textAlign = 'center';
    ctx.fillText(`-${data.latentRange}`, padLeft, totalH - 6);
    ctx.fillText('0', padLeft + (data.gridSize * cellSize) / 2, totalH - 6);
    ctx.fillText(`+${data.latentRange}`, padLeft + data.gridSize * cellSize, totalH - 6);
    ctx.fillText('latent dim 0', padLeft + (data.gridSize * cellSize) / 2, totalH - 20);

    // Y axis (dim 1)
    ctx.textAlign = 'right';
    ctx.fillText(`-${data.latentRange}`, padLeft - 4, padTop + 8);
    ctx.fillText(`+${data.latentRange}`, padLeft - 4, padTop + data.gridSize * cellSize);
    ctx.save();
    ctx.translate(12, padTop + (data.gridSize * cellSize) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('latent dim 1', 0, 0);
    ctx.restore();
  }, [data]);

  return (
    <div>
      <div className="confusion-scroll">
        <canvas ref={canvasRef} style={{ borderRadius: 4, border: '1px solid #45475a', display: 'block' }} />
      </div>
      <div className="heatmap-note" style={{ marginTop: 8 }}>
        Each cell is decoded from a different point in the 2D latent space.
        Smooth transitions mean the model learned a continuous, meaningful representation.
      </div>
    </div>
  );
}
