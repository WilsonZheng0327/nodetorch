// Shared canvas components used across multiple transformation visualizations.

import { useRef, useEffect } from 'react';
import type { FeatureMaps, HistogramData } from '../types';

/** Render a grid of feature map channels as grayscale canvases. */
export function FeatureMapsGrid({ data, label }: { data: FeatureMaps; label?: string }) {
  if (data.maps.length === 0) return <div className="tfm-empty">No data</div>;
  return (
    <div className="tfm-fmaps">
      {label && (
        <div className="tfm-fmaps-label">
          {label} &middot; {data.showing} of {data.channels} ch &middot; {data.height}&times;{data.width}
        </div>
      )}
      <div className="tfm-fmaps-grid">
        {data.maps.map((m, i) => (
          <GrayscaleCanvas key={i} pixels={m} label={`${i}`} />
        ))}
      </div>
    </div>
  );
}

/** Single grayscale image canvas with a label. */
export function GrayscaleCanvas({ pixels, label, size = 72 }: { pixels: number[][]; label?: string; size?: number }) {
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
        data.data[idx] = v; data.data[idx + 1] = v; data.data[idx + 2] = v; data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels]);
  return (
    <div className="tfm-fmap-item">
      <canvas ref={canvasRef} style={{ width: size, height: size, imageRendering: 'pixelated' }} className="tfm-fmap-canvas" />
      {label && <span className="tfm-fmap-label">{label}</span>}
    </div>
  );
}

/** Arrow between two panels. */
export function Arrow({ label }: { label?: string }) {
  return (
    <div className="tfm-arrow">
      <span className="tfm-arrow-line">&#x2192;</span>
      {label && <span className="tfm-arrow-label">{label}</span>}
    </div>
  );
}

/** Adaptive axis formatting. */
export function fmtAxis(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  if (abs >= 0.01) return v.toFixed(2);
  return v.toExponential(1);
}

/** Bar chart with Y-axis labels and gridlines.
 *  Always includes 0 and leaves padding above 0 even if all values are negative. */
export function VectorBars({ values, height = 180, label }: { values: number[]; height?: number; label?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || values.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    const yAxisW = 54;
    const padTop = 10;
    const padBot = 24;
    const plotW = W - yAxisW;
    const plotH = H - padTop - padBot;

    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    // Always include 0, and add ~15% padding above 0 if all values are on one side
    let vmin = Math.min(dataMin, 0);
    let vmax = Math.max(dataMax, 0);
    const dataRange = vmax - vmin || 1;
    if (dataMax <= 0) vmax = Math.abs(vmin) * 0.15;  // all negative: show some space above 0
    if (dataMin >= 0) vmin = -vmax * 0.08;            // all positive: show a sliver below 0
    const range = vmax - vmin || 1;

    ctx.clearRect(0, 0, W, H);

    // Y-axis ticks — evenly spaced, always including 0
    const nTicks = 5;
    const step = range / nTicks;
    const ticks: number[] = [];
    for (let i = 0; i <= nTicks; i++) ticks.push(vmin + i * step);
    // Snap a tick to 0 if close, or insert 0
    const zeroIdx = ticks.findIndex(t => Math.abs(t) < step * 0.15);
    if (zeroIdx >= 0) ticks[zeroIdx] = 0;
    else { ticks.push(0); ticks.sort((a, b) => a - b); }

    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (const val of ticks) {
      const y = padTop + ((vmax - val) / range) * plotH;
      ctx.strokeStyle = val === 0 ? '#585b70' : '#313244';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(yAxisW, y + 0.5);
      ctx.lineTo(W, y + 0.5);
      ctx.stroke();
      ctx.fillStyle = val === 0 ? '#a6adc8' : '#585b70';
      ctx.fillText(fmtAxis(val), yAxisW - 5, y + 4);
    }

    // Bars
    const bw = Math.max(1, plotW / values.length);
    const zeroY = padTop + ((vmax - 0) / range) * plotH;
    for (let i = 0; i < values.length; i++) {
      const x = yAxisW + i * bw;
      const valY = padTop + ((vmax - values[i]) / range) * plotH;
      const barTop = Math.min(zeroY, valY);
      const barH = Math.abs(zeroY - valY);
      ctx.fillStyle = values[i] >= 0 ? '#89b4fa' : '#f38ba8';
      ctx.globalAlpha = 0.7;
      ctx.fillRect(x, barTop, Math.max(1, bw - (bw > 3 ? 0.5 : 0)), barH);
    }
    ctx.globalAlpha = 1;

    // X-axis count
    ctx.fillStyle = '#585b70';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${values.length} values`, yAxisW + plotW / 2, H - 2);
  }, [values]);
  return (
    <div>
      {label && <div className="tfm-panel-label">{label}</div>}
      <canvas ref={canvasRef} className="tfm-vector-canvas" style={{ height }} />
    </div>
  );
}

/** Histogram chart. Accepts optional yMax for syncing Y axes across paired histograms. */
export function Histogram({ data, color = '#89b4fa', label, yMax }: {
  data: HistogramData; color?: string; label?: string; yMax?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data?.bins || data.bins.length === 0 || !data.counts || data.counts.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    const labelH = 22;
    const plotH = h - labelH;
    const maxCount = yMax ?? Math.max(...data.counts);
    if (maxCount === 0) return;
    const barWidth = w / data.counts.length;
    const vmin = data.bins[0];
    const vmax = data.bins[data.bins.length - 1] + (data.bins.length > 1 ? data.bins[1] - data.bins[0] : 1);
    const vrange = vmax - vmin || 1;

    ctx.clearRect(0, 0, w, h);

    // Bars — tallest bar touches the top
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < data.counts.length; i++) {
      const barH = (data.counts[i] / maxCount) * plotH;
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
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('0', zeroX, plotH + 12);
    }

    // Stats
    ctx.fillStyle = '#a6adc8';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`\u03BC=${fmtAxis(data.mean)}  \u03C3=${fmtAxis(data.std)}`, 2, h - 3);
    ctx.textAlign = 'right';
    ctx.fillText(`[${fmtAxis(vmin)}, ${fmtAxis(vmax)}]`, w - 2, h - 3);
  }, [data, color, yMax]);

  return (
    <div>
      {label && <div className="tfm-panel-label">{label}</div>}
      <canvas ref={canvasRef} className="tfm-histogram-canvas" />
    </div>
  );
}

/** 2D heatmap canvas (cool colormap). */
export function HeatmapCanvas({ data, rows, cols, min, max }: {
  data: number[][]; rows: number; cols: number; min: number; max: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const cellSize = Math.max(2, Math.min(6, Math.floor(300 / Math.max(rows, cols))));
    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const range = max - min || 1;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = data[r]?.[c] ?? 0;
        const t = Math.max(0, Math.min(1, (v - min) / range));
        const red = Math.round(t < 0.5 ? 0 : (t - 0.5) * 2 * 255);
        const green = Math.round(t < 0.5 ? t * 2 * 200 : 200 + (t - 0.5) * 2 * 55);
        const blue = Math.round(t < 0.5 ? 100 + t * 2 * 155 : 255 - (t - 0.5) * 2 * 255);
        ctx.fillStyle = `rgb(${red},${green},${blue})`;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }
  }, [data, rows, cols, min, max]);
  return <canvas ref={canvasRef} className="tfm-heatmap-canvas" />;
}
