// ExtraPanels — renders Stage.extras as additional panels in the detail view.
// Each kind has its own small component. New kinds can be added without touching others.

import { useRef, useEffect } from 'react';
import type { Extra, Stats } from './types';

export function ExtraPanels({ extras }: { extras: Extra[] }) {
  return (
    <div className="stage-extras">
      {extras.map((extra, i) => (
        <ExtraPanel key={i} extra={extra} />
      ))}
    </div>
  );
}

function ExtraPanel({ extra }: { extra: Extra }) {
  switch (extra.kind) {
    case 'before_after_histograms':
      return <BeforeAfterHistograms input={extra.input} output={extra.output} />;
    case 'conv_kernels':
      return <ConvKernels data={extra} />;
    case 'weight_matrix':
      return <WeightMatrix data={extra} />;
    case 'attention_map':
      return <AttentionMap data={extra} />;
    case 'recurrent_state':
      return <RecurrentState data={extra} />;
    case 'saliency_map':
      return <SaliencyMap data={extra} />;
    // Backward-specific extras
    case 'gradient_kernels':
      return <GradientKernels data={extra} />;
    case 'spatial_gradient_heatmap':
      return <SpatialGradientHeatmap data={extra} />;
    case 'per_unit_gradient_bars':
      return <PerUnitGradientBars data={extra} />;
    case 'gradient_weight_matrix':
      return <GradientWeightMatrix data={extra} />;
    default:
      return null;
  }
}

// --- Before/after histograms (normalization + activation layers) ---

function BeforeAfterHistograms({ input, output }: { input: Stats; output: Stats }) {
  return (
    <div className="extra-panel">
      <div className="extra-panel-title">Before / After</div>
      <div className="extra-histograms-pair">
        <HistogramWithAxes stats={input} label="Input" color="#6c7086" />
        <div className="extra-histograms-arrow">→</div>
        <HistogramWithAxes stats={output} label="Output" color="#89b4fa" />
      </div>
    </div>
  );
}

function HistogramWithAxes({ stats, label, color }: { stats: Stats; label: string; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !stats.histBins || !stats.histCounts || stats.histBins.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const labelH = 12;
    const plotH = h - labelH;
    const bins = stats.histBins;
    const counts = stats.histCounts;
    const maxCount = Math.max(...counts);
    if (maxCount === 0) return;
    const barWidth = w / counts.length;

    const vmin = bins[0];
    const vmax = bins[bins.length - 1] + (bins.length > 1 ? bins[1] - bins[0] : 1);
    const vrange = vmax - vmin || 1;

    ctx.clearRect(0, 0, w, h);

    // Bars
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < counts.length; i++) {
      const barH = (counts[i] / maxCount) * (plotH - 2);
      ctx.fillRect(i * barWidth, plotH - barH, barWidth - 0.5, barH);
    }
    ctx.globalAlpha = 1;

    // Zero line
    if (vmin < 0 && vmax > 0) {
      const zeroX = ((0 - vmin) / vrange) * w;
      ctx.strokeStyle = '#cdd6f4';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(zeroX, 0);
      ctx.lineTo(zeroX, plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#cdd6f4';
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('0', zeroX, h - 1);
    }

    // Min/max
    ctx.fillStyle = '#585b70';
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(fmt(vmin), 1, h - 1);
    ctx.textAlign = 'right';
    ctx.fillText(fmt(vmax), w - 1, h - 1);
  }, [stats, color]);

  return (
    <div className="extra-histogram-wrap">
      <div className="extra-histogram-label">{label}</div>
      <canvas ref={canvasRef} className="extra-histogram-canvas" />
      <div className="extra-histogram-stats">
        μ={fmt(stats.mean ?? 0)}  σ={fmt(stats.std ?? 0)}
      </div>
    </div>
  );
}

// --- Conv2d kernels ---

function ConvKernels({ data }: { data: Extract<Extra, { kind: 'conv_kernels' }> }) {
  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Learned Kernels ({data.showing} of {data.totalFilters} filters, {data.kernelHeight}×{data.kernelWidth}, averaged across {data.inChannels}{data.inChannels === 1 ? ' channel' : ' channels'})
      </div>
      <div className="extra-kernel-grid">
        {data.kernels.map((k, i) => (
          <KernelCanvas key={i} pixels={k} label={`f${i}`} />
        ))}
      </div>
    </div>
  );
}

function KernelCanvas({ pixels, label }: { pixels: number[][]; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length;
    const w = pixels[0].length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const v = pixels[y][x];
        data.data[idx] = v;
        data.data[idx + 1] = v;
        data.data[idx + 2] = v;
        data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels]);
  return (
    <div className="extra-kernel-item">
      <canvas ref={canvasRef} className="extra-kernel-canvas" />
      <span className="extra-kernel-label">{label}</span>
    </div>
  );
}

// --- Linear weight matrix ---

function WeightMatrix({ data }: { data: Extract<Extra, { kind: 'weight_matrix' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cellSize = Math.max(2, Math.min(6, Math.floor(400 / Math.max(data.rows, data.cols))));
    canvas.width = data.cols * cellSize;
    canvas.height = data.rows * cellSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = data.max - data.min || 1;
    for (let r = 0; r < data.rows; r++) {
      for (let c = 0; c < data.cols; c++) {
        const v = data.data[r]?.[c] ?? 0;
        const t = (v - data.min) / range;
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
  }, [data]);

  const downsampled = data.rows < data.actualRows || data.cols < data.actualCols;

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Weight Matrix ({data.actualRows} × {data.actualCols})
        {downsampled && <span className="extra-panel-note"> — downsampled to {data.rows} × {data.cols}</span>}
      </div>
      <canvas ref={canvasRef} className="extra-weight-matrix" />
      <div className="extra-weight-legend">
        <span>{data.min.toFixed(3)}</span>
        <div className="extra-weight-legend-bar" />
        <span>{data.max.toFixed(3)}</span>
      </div>
    </div>
  );
}

// --- Attention map (heatmap) ---

function AttentionMap({ data }: { data: Extract<Extra, { kind: 'attention_map' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cellSize = Math.max(3, Math.min(10, Math.floor(400 / Math.max(data.rows, data.cols))));
    canvas.width = data.cols * cellSize;
    canvas.height = data.rows * cellSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    for (let r = 0; r < data.rows; r++) {
      for (let c = 0; c < data.cols; c++) {
        const v = data.data[r]?.[c] ?? 0;
        // Attention weights are in [0, 1] already
        ctx.fillStyle = warmColor(Math.max(0, Math.min(1, v)));
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
  }, [data]);

  const downsampled = data.rows < data.actualRows || data.cols < data.actualCols;

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Attention Map ({data.actualRows} × {data.actualCols})
        {downsampled && <span className="extra-panel-note"> — downsampled to {data.rows} × {data.cols}</span>}
      </div>
      <div className="extra-attention-wrap">
        <div className="extra-attention-axis-y">query ↓</div>
        <div>
          <canvas ref={canvasRef} className="extra-attention-canvas" />
          <div className="extra-attention-axis-x">key →</div>
        </div>
      </div>
      <div className="extra-weight-legend">
        <span>0</span>
        <div className="extra-weight-legend-bar extra-attention-legend-bar" />
        <span>1</span>
      </div>
    </div>
  );
}

// --- Recurrent state (hidden/cell vector) ---

function RecurrentState({ data }: { data: Extract<Extra, { kind: 'recurrent_state' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.values.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    const min = Math.min(0, ...data.values);
    const max = Math.max(0, ...data.values);
    const range = max - min || 1;
    ctx.clearRect(0, 0, w, h);
    const bw = w / data.values.length;
    const zeroY = h - ((0 - min) / range) * h;

    for (let i = 0; i < data.values.length; i++) {
      const v = data.values[i];
      const bh = Math.abs(v / range) * h;
      const y = v >= 0 ? zeroY - bh : zeroY;
      ctx.fillStyle = v >= 0 ? '#89b4fa' : '#f38ba8';
      ctx.globalAlpha = 0.8;
      ctx.fillRect(i * bw, y, Math.max(1, bw - 0.3), bh);
    }
    ctx.globalAlpha = 1;

    // Zero line
    ctx.strokeStyle = '#6c7086';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [data]);

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        {data.label} ({data.totalLength} units)
      </div>
      <canvas ref={canvasRef} className="extra-recurrent-canvas" />
    </div>
  );
}

// --- Saliency map ---

function SaliencyMap({ data }: { data: Extract<Extra, { kind: 'saliency_map' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.pixels.length === 0) return;
    canvas.width = data.width;
    canvas.height = data.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(data.width, data.height);
    for (let y = 0; y < data.height; y++) {
      for (let x = 0; x < data.width; x++) {
        const idx = (y * data.width + x) * 4;
        const v = data.pixels[y][x] ?? 0;
        // Warm heatmap: low = black, high = bright yellow
        const t = v / 255;
        img.data[idx] = Math.round(Math.min(255, t * 2 * 255));
        img.data[idx + 1] = Math.round(Math.max(0, (t - 0.5) * 2 * 255));
        img.data[idx + 2] = 0;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Saliency Map — gradient of predicted class w.r.t. input
        <span className="extra-panel-note"> &middot; predicted class {data.predictedClass}</span>
      </div>
      <div className="extra-saliency-desc">
        Bright pixels = model's decision depends on them heavily.
      </div>
      <canvas ref={canvasRef} className="extra-saliency-canvas" />
    </div>
  );
}

// --- Backward-specific extras ---

function GradientKernels({ data }: { data: Extract<Extra, { kind: 'gradient_kernels' }> }) {
  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Gradient Kernels ({data.showing} of {data.totalFilters} filters, {data.kernelHeight}×{data.kernelWidth})
      </div>
      <div className="extra-kernel-grid">
        {data.kernels.map((k, i) => (
          <KernelCanvas key={i} pixels={k} label={`f${i}`} />
        ))}
      </div>
    </div>
  );
}

function SpatialGradientHeatmap({ data }: { data: Extract<Extra, { kind: 'spatial_gradient_heatmap' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.pixels.length === 0) return;
    canvas.width = data.width;
    canvas.height = data.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(data.width, data.height);
    for (let y = 0; y < data.height; y++) {
      for (let x = 0; x < data.width; x++) {
        const idx = (y * data.width + x) * 4;
        const v = data.pixels[y][x] ?? 0;
        const t = v / 255;
        img.data[idx] = Math.round(Math.min(255, t * 2 * 255));
        img.data[idx + 1] = Math.round(Math.max(0, (t - 0.5) * 2 * 255));
        img.data[idx + 2] = 0;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [data]);

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Spatial Gradient Heatmap — where gradient signal is strongest
      </div>
      <canvas ref={canvasRef} className="extra-saliency-canvas" />
    </div>
  );
}

function PerUnitGradientBars({ data }: { data: Extract<Extra, { kind: 'per_unit_gradient_bars' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.values.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    const max = Math.max(...data.values.map(Math.abs), 1e-8);
    ctx.clearRect(0, 0, w, h);
    const bw = w / data.values.length;
    for (let i = 0; i < data.values.length; i++) {
      const x = Math.round(i * bw);
      const nextX = Math.round((i + 1) * bw);
      const bh = Math.round((Math.abs(data.values[i]) / max) * (h - 1));
      ctx.fillStyle = data.values[i] >= 0 ? '#f9e2af' : '#f38ba8';
      ctx.globalAlpha = 0.8;
      ctx.fillRect(x, h - bh, Math.max(1, nextX - x - 1), bh);
    }
    ctx.globalAlpha = 1;
  }, [data]);

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">{data.label}</div>
      <canvas ref={canvasRef} className="extra-recurrent-canvas" />
    </div>
  );
}

function GradientWeightMatrix({ data }: { data: Extract<Extra, { kind: 'gradient_weight_matrix' }> }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cellSize = Math.max(2, Math.min(6, Math.floor(400 / Math.max(data.rows, data.cols))));
    canvas.width = data.cols * cellSize;
    canvas.height = data.rows * cellSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = data.max - data.min || 1;
    for (let r = 0; r < data.rows; r++) {
      for (let c = 0; c < data.cols; c++) {
        const v = data.data[r]?.[c] ?? 0;
        const t = (v - data.min) / range;
        ctx.fillStyle = heatColor(t);
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
  }, [data]);

  const downsampled = data.rows < data.actualRows || data.cols < data.actualCols;

  return (
    <div className="extra-panel">
      <div className="extra-panel-title">
        Weight Gradient Matrix ({data.actualRows} × {data.actualCols})
        {downsampled && <span className="extra-panel-note"> — downsampled to {data.rows} × {data.cols}</span>}
      </div>
      <canvas ref={canvasRef} className="extra-weight-matrix" />
      <div className="extra-weight-legend">
        <span>{data.min.toFixed(3)}</span>
        <div className="extra-weight-legend-bar" />
        <span>{data.max.toFixed(3)}</span>
      </div>
    </div>
  );
}

// --- Utilities ---

function fmt(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}

function heatColor(t: number): string {
  t = Math.max(0, Math.min(1, t));
  // cool: dark blue → cyan → green → yellow
  const r = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
  const g = Math.round(t < 0.5 ? t * 2 * 200 : 200 + (t - 0.5) * 2 * 55);
  const b = Math.round(t < 0.5 ? 100 + t * 2 * 155 : 255 - (t - 0.5) * 2 * 255);
  return `rgb(${r},${g},${b})`;
}

function warmColor(t: number): string {
  t = Math.max(0, Math.min(1, t));
  // Black → Red → Yellow → White
  const r = Math.round(Math.min(255, t * 2 * 255));
  const g = Math.round(Math.max(0, (t - 0.5) * 2 * 255));
  const b = Math.round(Math.max(0, (t - 0.75) * 4 * 255));
  return `rgb(${r},${g},${b})`;
}
