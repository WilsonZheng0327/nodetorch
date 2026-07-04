// Canvas charts for the training dashboard: the loss/accuracy line chart, the
// gradient charts (per-layer depth view + per-layer trend over epochs), and the
// per-class accuracy bars. Styles come from TrainingDashboard.css.

import { useRef, useEffect, useState } from 'react';

// --- Canvas line chart (loss / accuracy, with optional val + compare series) ---

interface ChartProps {
  data: number[];
  labels: number[];
  color: string;
  formatValue: (v: number) => string;
  selectedIndex?: number | null;
  valData?: (number | null | undefined)[];
  valColor?: string;
  compareData?: (number | null | undefined)[];
  compareColor?: string;
  compareLabel?: string;
}

export function Chart({
  data: rawData,
  labels,
  color,
  formatValue,
  selectedIndex,
  valData,
  valColor,
  compareData,
  compareColor,
  compareLabel,
}: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Track canvas client size so we re-render on resize (keeps bitmap crisp)
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => setSize({ w: canvas.clientWidth, h: canvas.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rawData.length === 0 || size.w === 0) return;

    // Guard against NaN/null/Infinity
    const data = rawData.map((v) => (v == null || !isFinite(v) ? 0 : v));
    // Sanitize valData (preserve null = "no val for this epoch")
    const valClean = valData?.map((v) => (v != null && isFinite(v) ? v : null));
    const compareClean = compareData?.map((v) => (v != null && isFinite(v) ? v : null));

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    // Use clientWidth/Height (excludes padding) for accurate bitmap sizing.
    const w = size.w;
    const h = size.h;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { top: 40, right: 12, bottom: 28, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Compute min/max across all series
    const valValuesForRange = valClean?.filter((v): v is number => v != null) ?? [];
    const compareValuesForRange = compareClean?.filter((v): v is number => v != null) ?? [];
    const allValues = [...data, ...valValuesForRange, ...compareValuesForRange];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    // Y axis labels
    ctx.fillStyle = '#6c7086';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = min + (range * i) / 4;
      const y = pad.top + plotH - (plotH * i) / 4;
      ctx.fillText(formatValue(val), pad.left - 6, y + 3);

      ctx.strokeStyle = '#313244';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // X axis labels
    ctx.textAlign = 'center';
    ctx.fillStyle = '#6c7086';
    const step = Math.max(1, Math.floor(data.length / 6));
    for (let i = 0; i < data.length; i += step) {
      const x = pad.left + (plotW * i) / Math.max(data.length - 1, 1);
      ctx.fillText(String(labels[i]), x, h - 4);
    }

    // Data line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (plotW * i) / Math.max(data.length - 1, 1);
      const y = pad.top + plotH - (plotH * (data[i] - min)) / range;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Data points
    ctx.fillStyle = color;
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (plotW * i) / Math.max(data.length - 1, 1);
      const y = pad.top + plotH - (plotH * (data[i] - min)) / range;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Validation line (dashed) if provided
    if (valClean) {
      const vColor = valColor ?? '#fab387';
      ctx.strokeStyle = vColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < valClean.length; i++) {
        const v = valClean[i];
        if (v == null) {
          started = false;
          continue;
        }
        const x = pad.left + (plotW * i) / Math.max(valClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // Val points
      ctx.fillStyle = vColor;
      for (let i = 0; i < valClean.length; i++) {
        const v = valClean[i];
        if (v == null) continue;
        const x = pad.left + (plotW * i) / Math.max(valClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Compare line (dotted, purple by default)
    if (compareClean && compareClean.length > 0) {
      const cColor = compareColor ?? '#cba6f7';
      ctx.strokeStyle = cColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([1, 3]);
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < compareClean.length; i++) {
        const v = compareClean[i];
        if (v == null) {
          started = false;
          continue;
        }
        const x = pad.left + (plotW * i) / Math.max(compareClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // Legend (if val line present). Sits in the top margin with breathing room above.
    if (valClean) {
      const legendY = 18; // text baseline
      const legendRectY = 12; // line sample top
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'left';
      // Train legend
      ctx.fillStyle = color;
      ctx.fillRect(pad.left + 4, legendRectY, 10, 2);
      ctx.fillStyle = '#a6adc8';
      ctx.fillText('train', pad.left + 18, legendY);
      // Val legend
      ctx.strokeStyle = valColor ?? '#fab387';
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(pad.left + 60, legendRectY + 1);
      ctx.lineTo(pad.left + 70, legendRectY + 1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#a6adc8';
      ctx.fillText('val', pad.left + 74, legendY);
      // Compare legend
      if (compareClean && compareClean.length > 0) {
        ctx.strokeStyle = compareColor ?? '#cba6f7';
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(pad.left + 110, legendRectY + 1);
        ctx.lineTo(pad.left + 120, legendRectY + 1);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#cba6f7';
        const labelText = compareLabel && compareLabel.length < 40 ? compareLabel : 'compare';
        ctx.fillText(labelText, pad.left + 124, legendY);
      }
    }

    // Epoch marker: a dotted vertical line + dot at the epoch in view. When
    // scrubbing history it's blue at the selected epoch; when viewing the latest
    // epoch (selectedIndex == null) it's orange at the final point to signal
    // "current / latest".
    const isCurrentEpoch = selectedIndex == null;
    const markerIndex = isCurrentEpoch
      ? data.length - 1
      : selectedIndex >= 0 && selectedIndex < data.length
        ? selectedIndex
        : -1;
    if (markerIndex >= 0) {
      const markerColor = isCurrentEpoch ? '#fe640b' : '#89b4fa';
      const sx = pad.left + (plotW * markerIndex) / Math.max(data.length - 1, 1);
      const sy = pad.top + plotH - (plotH * (data[markerIndex] - min)) / range;
      // Vertical line
      ctx.strokeStyle = markerColor;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx, pad.top);
      ctx.lineTo(sx, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      // Highlight dot
      ctx.fillStyle = markerColor;
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fill();
      // Value label
      ctx.fillStyle = '#cdd6f4';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(formatValue(data[markerIndex]), sx, sy - 10);
    }
  }, [
    rawData,
    labels,
    color,
    formatValue,
    selectedIndex,
    valData,
    valColor,
    compareData,
    compareColor,
    compareLabel,
    size,
  ]);

  return <canvas ref={canvasRef} className="dashboard-chart" />;
}

// --- Gradient flow horizontal bar chart ---

// Per-layer gradient magnitude (RMS = norm / sqrt(params), so layer sizes are
// comparable). `norm` is the L2 norm fallback for older data without rms.
type GradEntry = { name: string; norm: number; rms?: number };

const gradMag = (e: GradEntry) => e.rms ?? e.norm;

// Distinct line colors for the per-layer trend chart (Catppuccin accents).
const LAYER_COLORS = [
  '#89b4fa',
  '#a6e3a1',
  '#fab387',
  '#f38ba8',
  '#cba6f7',
  '#94e2d5',
  '#f9e2af',
  '#eba0ac',
  '#74c7ec',
  '#b4befe',
];

/**
 * Shared log10 y-domain across the whole run, so scrubbing the depth chart and
 * reading the trend chart both use one fixed absolute scale (no per-epoch
 * renormalization). Returns [loExp, hiExp] as integer powers of 10, or null.
 */
export function gradientLogDomain(
  progress: { gradientFlow?: GradEntry[] }[],
): [number, number] | null {
  let lo = Infinity;
  let hi = -Infinity;
  for (const ep of progress) {
    for (const e of ep.gradientFlow ?? []) {
      const m = gradMag(e);
      if (m == null || !isFinite(m) || m <= 0) continue;
      lo = Math.min(lo, m);
      hi = Math.max(hi, m);
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) return null;
  let loExp = Math.floor(Math.log10(lo));
  let hiExp = Math.ceil(Math.log10(hi));
  // Guarantee at least 3 decades so a tight cluster still reads as absolute.
  while (hiExp - loExp < 3) {
    loExp -= 1;
    hiExp += 1;
  }
  return [loExp, hiExp];
}

function drawLogGrid(
  ctx: CanvasRenderingContext2D,
  plot: { x: number; y: number; w: number; h: number },
  domain: [number, number],
) {
  const [loExp, hiExp] = domain;
  const yFor = (exp: number) => plot.y + plot.h - ((exp - loExp) / (hiExp - loExp)) * plot.h;
  ctx.font = '10px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let exp = loExp; exp <= hiExp; exp++) {
    const y = yFor(exp);
    ctx.strokeStyle = exp === 0 ? '#585b70' : '#313244';
    ctx.beginPath();
    ctx.moveTo(plot.x, y);
    ctx.lineTo(plot.x + plot.w, y);
    ctx.stroke();
    ctx.fillStyle = '#6c7086';
    ctx.fillText(`1e${exp}`, plot.x - 6, y);
  }
}

// --- Graph 1: gradient magnitude across layers (one epoch, log scale) ---

export function GradientDepthChart({
  data,
  domain,
}: {
  data: GradEntry[];
  domain: [number, number] | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => setSize({ w: canvas.clientWidth, h: canvas.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0 || size.w === 0 || !domain) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = size.w;
    const h = size.h;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const [loExp, hiExp] = domain;
    const plot = { x: 44, y: 8, w: w - 56, h: h - 40 };
    drawLogGrid(ctx, plot, domain);

    const n = data.length;
    const xFor = (i: number) => plot.x + (n === 1 ? plot.w / 2 : (i / (n - 1)) * plot.w);
    const yFor = (m: number) => {
      const exp = m > 0 ? Math.log10(m) : loExp;
      const clamped = Math.max(loExp, Math.min(hiExp, exp));
      return plot.y + plot.h - ((clamped - loExp) / (hiExp - loExp)) * plot.h;
    };

    // Connecting line
    ctx.strokeStyle = '#89b4fa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((e, i) => {
      const x = xFor(i);
      const y = yFor(gradMag(e));
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Points + x labels. With many layers, thin the labels (~8 max) and skip
    // the per-point dots, but always flag dead (zero) gradients in red.
    const labelStep = Math.max(1, Math.ceil(n / 8));
    const showDots = n <= 24;
    ctx.font = '9px Inter, system-ui, sans-serif';
    data.forEach((e, i) => {
      const x = xFor(i);
      const m = gradMag(e);
      const dead = !(m > 0);
      const y = yFor(m);
      if (showDots || dead) {
        ctx.fillStyle = dead ? '#f38ba8' : '#89b4fa';
        ctx.beginPath();
        ctx.arc(x, y, dead ? 3 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // rotated layer label (thinned when crowded)
      if (i % labelStep === 0 || i === n - 1) {
        ctx.save();
        ctx.translate(x, plot.y + plot.h + 6);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = '#6c7086';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        let label = e.name;
        if (ctx.measureText(label).width > 60) {
          while (label.length > 3 && ctx.measureText(label + '…').width > 60)
            label = label.slice(0, -1);
          label += '…';
        }
        ctx.fillText(label, 0, 0);
        ctx.restore();
      }
    });
  }, [data, size, domain]);

  if (data.length === 0) {
    return (
      <div className="dashboard-chart-placeholder">No gradient data yet — train to see results</div>
    );
  }

  return (
    <>
      <div className="dashboard-explainer">
        Per-layer gradient magnitude (RMS) for this epoch, input → output, on a log scale. A
        downhill slope toward the input means vanishing gradients; an uphill blow-up means exploding
        gradients. Red points are dead (zero) gradients. Scrub epochs to watch the profile change.
      </div>
      <canvas ref={canvasRef} className="dashboard-chart grad-chart" />
    </>
  );
}

// --- Graph 2: each layer's gradient magnitude across epochs (log scale) ---

export function GradientTrendChart({
  progress,
  domain,
  selectedEpoch,
}: {
  progress: { epoch: number; gradientFlow?: GradEntry[] }[];
  domain: [number, number] | null;
  selectedEpoch?: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  // Build one series per layer name, aligned across epochs.
  const layerNames =
    progress.find((p) => p.gradientFlow?.length)?.gradientFlow?.map((e) => e.name) ?? [];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => setSize({ w: canvas.clientWidth, h: canvas.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w === 0 || !domain || layerNames.length === 0 || progress.length === 0)
      return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = size.w;
    const h = size.h;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const [loExp, hiExp] = domain;
    const plot = { x: 44, y: 8, w: w - 56, h: h - 24 };
    drawLogGrid(ctx, plot, domain);

    const epochs = progress.map((p) => p.epoch);
    const eMin = Math.min(...epochs);
    const eMax = Math.max(...epochs);
    const xFor = (ep: number) =>
      plot.x + (eMax === eMin ? plot.w / 2 : ((ep - eMin) / (eMax - eMin)) * plot.w);
    const yFor = (m: number) => {
      const exp = m > 0 ? Math.log10(m) : loExp;
      const clamped = Math.max(loExp, Math.min(hiExp, exp));
      return plot.y + plot.h - ((clamped - loExp) / (hiExp - loExp)) * plot.h;
    };

    layerNames.forEach((name, li) => {
      ctx.strokeStyle = LAYER_COLORS[li % LAYER_COLORS.length];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      progress.forEach((p) => {
        const entry = p.gradientFlow?.find((e) => e.name === name);
        if (!entry) return;
        const x = xFor(p.epoch);
        const y = yFor(gradMag(entry));
        started ? ctx.lineTo(x, y) : ((started = true), ctx.moveTo(x, y));
      });
      ctx.stroke();
    });

    // Epoch indicator: dotted vertical line at the epoch in view, following the
    // slider. Orange at the latest epoch (selectedEpoch == null), blue when
    // scrubbing — same convention as the loss/accuracy chart.
    const isCurrent = selectedEpoch == null;
    const markerIdx = isCurrent
      ? progress.length - 1
      : Math.min(Math.max(selectedEpoch - 1, 0), progress.length - 1);
    const markerX = xFor(progress[markerIdx].epoch);
    ctx.strokeStyle = isCurrent ? '#fe640b' : '#89b4fa';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(markerX, plot.y);
    ctx.lineTo(markerX, plot.y + plot.h);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [progress, size, domain, layerNames, selectedEpoch]);

  if (layerNames.length === 0) {
    return (
      <div className="dashboard-chart-placeholder">
        No gradient history yet — train to see results
      </div>
    );
  }

  return (
    <>
      <div className="dashboard-explainer">
        Each layer's gradient magnitude (RMS) across epochs, log scale. Lines drifting toward the
        floor signal vanishing gradients; lines climbing signal instability. Ideally they stay
        within a stable band.
      </div>
      <canvas ref={canvasRef} className="dashboard-chart grad-chart" />
      <div className="grad-legend">
        {layerNames.map((name, li) => (
          <span key={name} className="grad-legend-item">
            <span
              className="grad-legend-swatch"
              style={{ background: LAYER_COLORS[li % LAYER_COLORS.length] }}
            />
            {name}
          </span>
        ))}
      </div>
    </>
  );
}

// Thermal colormap: dark (vanishing) → blue → teal → amber → red (exploding).
const HEAT_STOPS: [number, [number, number, number]][] = [
  [0.0, [24, 24, 37]],
  [0.25, [49, 78, 173]],
  [0.5, [46, 160, 133]],
  [0.75, [249, 176, 0]],
  [1.0, [240, 62, 62]],
];

function heatColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    const [t1, c1] = HEAT_STOPS[i];
    if (x <= t1) {
      const [t0, c0] = HEAT_STOPS[i - 1];
      const f = (x - t0) / (t1 - t0 || 1);
      return `rgb(${Math.round(c0[0] + (c1[0] - c0[0]) * f)}, ${Math.round(
        c0[1] + (c1[1] - c0[1]) * f,
      )}, ${Math.round(c0[2] + (c1[2] - c0[2]) * f)})`;
    }
  }
  const last = HEAT_STOPS[HEAT_STOPS.length - 1][1];
  return `rgb(${last[0]}, ${last[1]}, ${last[2]})`;
}

// --- Graph 3: gradient magnitude heatmap (layers × epochs) — scales to deep nets ---

export function GradientHeatmap({
  progress,
  domain,
  selectedEpoch,
}: {
  progress: { epoch: number; gradientFlow?: GradEntry[] }[];
  domain: [number, number] | null;
  selectedEpoch?: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  const layerNames =
    progress.find((p) => p.gradientFlow?.length)?.gradientFlow?.map((e) => e.name) ?? [];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => setSize({ w: canvas.clientWidth, h: canvas.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.w === 0 || !domain || layerNames.length === 0 || progress.length === 0)
      return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = size.w;
    const h = size.h;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const [loExp, hiExp] = domain;
    const nLayers = layerNames.length;
    const nEpochs = progress.length;

    const grid = { x: 92, y: 4, w: w - 92 - 64, h: h - 4 - 20 };
    const cellW = grid.w / nEpochs;
    const cellH = grid.h / nLayers;

    // Cells: row = layer (input at top), col = epoch, color = normalized log magnitude.
    for (let li = 0; li < nLayers; li++) {
      const name = layerNames[li];
      for (let ei = 0; ei < nEpochs; ei++) {
        const entry = progress[ei].gradientFlow?.find((e) => e.name === name);
        if (!entry) {
          ctx.fillStyle = '#181825';
        } else {
          const m = gradMag(entry);
          const exp = m > 0 ? Math.log10(m) : loExp;
          ctx.fillStyle = heatColor(
            (Math.max(loExp, Math.min(hiExp, exp)) - loExp) / (hiExp - loExp),
          );
        }
        ctx.fillRect(grid.x + ei * cellW, grid.y + li * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Layer labels (rows), thinned to ~12px each minimum
    ctx.font = '9px Inter, system-ui, sans-serif';
    ctx.fillStyle = '#a6adc8';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const rowStep = Math.max(1, Math.ceil((nLayers * 12) / grid.h));
    for (let li = 0; li < nLayers; li++) {
      if (li % rowStep !== 0 && li !== nLayers - 1) continue;
      let label = layerNames[li];
      if (ctx.measureText(label).width > 84) {
        while (label.length > 3 && ctx.measureText(label + '…').width > 84)
          label = label.slice(0, -1);
        label += '…';
      }
      ctx.fillText(label, 86, grid.y + (li + 0.5) * cellH);
    }

    // Epoch labels (columns), thinned
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#6c7086';
    const colStep = Math.max(1, Math.ceil(nEpochs / 8));
    for (let ei = 0; ei < nEpochs; ei++) {
      if (ei % colStep !== 0 && ei !== nEpochs - 1) continue;
      ctx.fillText(String(progress[ei].epoch), grid.x + (ei + 0.5) * cellW, grid.y + grid.h + 4);
    }

    // Selected-epoch column outline (follows the slider; orange latest / blue scrubbing)
    const isCurrent = selectedEpoch == null;
    const selIdx = isCurrent ? nEpochs - 1 : Math.min(Math.max(selectedEpoch - 1, 0), nEpochs - 1);
    ctx.strokeStyle = isCurrent ? '#fe640b' : '#89b4fa';
    ctx.lineWidth = 2;
    ctx.strokeRect(grid.x + selIdx * cellW, grid.y, cellW, grid.h);

    // Colorbar
    const barX = w - 48;
    const barW = 12;
    const steps = 40;
    for (let s = 0; s < steps; s++) {
      ctx.fillStyle = heatColor(1 - s / (steps - 1));
      ctx.fillRect(barX, grid.y + (s / steps) * grid.h, barW, Math.ceil(grid.h / steps));
    }
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = '#6c7086';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let exp = loExp; exp <= hiExp; exp++) {
      const yy = grid.y + grid.h - ((exp - loExp) / (hiExp - loExp)) * grid.h;
      ctx.fillText(`1e${exp}`, barX + barW + 4, yy);
    }
  }, [progress, size, domain, layerNames, selectedEpoch]);

  if (layerNames.length === 0) {
    return (
      <div className="dashboard-chart-placeholder">
        No gradient history yet — train to see results
      </div>
    );
  }

  return (
    <>
      <div className="dashboard-explainer">
        Gradient magnitude (RMS) per layer (rows, input → output) across epochs (columns), log-color
        scale. Dark rows = vanishing gradients, red = exploding. The outlined column follows the
        epoch slider. Scales to deep networks where per-layer lines would overlap.
      </div>
      <canvas ref={canvasRef} className="dashboard-chart grad-chart" />
    </>
  );
}

// --- Per-class accuracy bar chart (all classes, sorted worst-first, scrollable) ---

export function PerClassChart({ data }: { data: { cls: number; accuracy: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const barH = 24;
    const gap = 3;
    const labelW = 48;
    const pad = { top: 8, right: 60, bottom: 8 };
    const totalH = pad.top + data.length * (barH + gap) + pad.bottom;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = totalH * dpr;
    canvas.style.height = `${totalH}px`;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const plotW = w - labelW - pad.right;

    ctx.clearRect(0, 0, w, totalH);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + gap);
      const barW = data[i].accuracy * plotW;

      ctx.fillStyle = '#a6adc8';
      ctx.font = '13px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(data[i].cls), labelW - 6, y + barH / 2 + 4);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const acc = data[i].accuracy;
      const r = Math.round((1 - acc) * 230);
      const g = Math.round(acc * 200);
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      ctx.fillStyle = '#cdd6f4';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${(acc * 100).toFixed(1)}%`, labelW + barW + 6, y + barH / 2 + 4);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No per-class data yet</div>;
  }

  return (
    <div className="dashboard-perclass-scroll">
      <canvas ref={canvasRef} className="dashboard-chart grad-chart" />
    </div>
  );
}
