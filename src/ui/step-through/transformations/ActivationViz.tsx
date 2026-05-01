// Activation transformation:
// - Activation curve diagram (separate, not overlaid on histogram)
// - Before/after feature maps (if CNN context)
// - Before/after histograms with shared X range and synced Y axis

import { useRef, useEffect } from 'react';
import type { ActivationTransformation, HistogramData } from '../types';
import { FeatureMapsGrid, Histogram, fmtAxis } from './shared';

function getCurve(fn: string, slope?: number): (x: number) => number {
  switch (fn) {
    case 'relu': return (x) => Math.max(0, x);
    case 'sigmoid': return (x) => 1 / (1 + Math.exp(-x));
    case 'tanh': return (x) => Math.tanh(x);
    case 'gelu': return (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
    case 'leaky_relu': { const s = slope ?? 0.01; return (x) => x >= 0 ? x : s * x; }
    default: return (x) => x;
  }
}

function getLabel(fn: string, slope?: number): string {
  switch (fn) {
    case 'relu': return 'ReLU: y = max(0, x)';
    case 'sigmoid': return 'Sigmoid: y = 1/(1+e\u207Bx)';
    case 'tanh': return 'Tanh: y = tanh(x)';
    case 'gelu': return 'GELU';
    case 'leaky_relu': return `Leaky ReLU: y = max(${slope ?? 0.01}x, x)`;
    default: return fn;
  }
}

export function ActivationViz({ t }: { t: ActivationTransformation }) {
  const hasMaps = t.inputMaps || t.outputMaps;
  const hasHist = t.inputHist?.bins && t.outputHist?.bins;

  const xMin = t.sharedXMin ?? (hasHist ? t.inputHist!.bins[0] : -1);
  const xMax = t.sharedXMax ?? (hasHist ? t.inputHist!.bins[t.inputHist!.bins.length - 1] : 1);
  const yMax = hasHist ? Math.max(...t.inputHist!.counts, ...t.outputHist!.counts) : 1;

  return (
    <div className="tfm-activation">
      {/* Activation function curve — standalone diagram */}
      <div className="tfm-section">
        <div className="tfm-section-title">{getLabel(t.fn, t.negativeSlope)}</div>
        <ActivationCurve fn={t.fn} xMin={xMin} xMax={xMax} negativeSlope={t.negativeSlope} />
      </div>

      {/* Before / After feature maps */}
      {hasMaps && (
        <div className="tfm-before-after">
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">Before</div>
            {t.inputMaps ? <FeatureMapsGrid data={t.inputMaps} /> : <div className="tfm-empty">&mdash;</div>}
          </div>
          <div className="tfm-ba-divider" />
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">After</div>
            {t.outputMaps ? <FeatureMapsGrid data={t.outputMaps} /> : <div className="tfm-empty">&mdash;</div>}
          </div>
        </div>
      )}

      {/* Before/after histograms — shared X range, synced Y */}
      {hasHist && (
        <div className="tfm-before-after">
          <div className="tfm-ba-pane">
            <FixedRangeHistogram data={t.inputHist!} color="#6c7086" label="Before" xMin={xMin} xMax={xMax} yMax={yMax} />
          </div>
          <div className="tfm-ba-divider" />
          <div className="tfm-ba-pane">
            <FixedRangeHistogram data={t.outputHist!} color="#89b4fa" label="After" xMin={xMin} xMax={xMax} yMax={yMax} />
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="tfm-activation-stats">
        {t.deadFraction != null && (
          <div className="tfm-stat">
            <span className="tfm-stat-value" style={{ color: '#f38ba8' }}>{(t.deadFraction * 100).toFixed(0)}%</span>
            <span className="tfm-stat-label">values zeroed</span>
          </div>
        )}
        {t.saturatedFraction != null && (
          <div className="tfm-stat">
            <span className="tfm-stat-value" style={{ color: '#f9e2af' }}>{(t.saturatedFraction * 100).toFixed(0)}%</span>
            <span className="tfm-stat-label">in saturated region</span>
          </div>
        )}
      </div>
    </div>
  );
}

/** Standalone activation curve diagram — shows the function shape with proper axes. */
function ActivationCurve({ fn, xMin, xMax, negativeSlope }: { fn: string; xMin: number; xMax: number; negativeSlope?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    const padL = 44;
    const padR = 10;
    const padT = 10;
    const padB = 24;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const curveFn = getCurve(fn, negativeSlope);

    // Compute Y range from curve output
    const yVals: number[] = [];
    for (let i = 0; i <= 100; i++) {
      yVals.push(curveFn(xMin + (i / 100) * (xMax - xMin)));
    }
    let yMin = Math.min(...yVals, 0);
    let yMax = Math.max(...yVals, 0);
    // Add a bit of padding
    const yPad = (yMax - yMin) * 0.08 || 0.1;
    yMin -= yPad;
    yMax += yPad;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const toX = (v: number) => padL + ((v - xMin) / xRange) * plotW;
    const toY = (v: number) => padT + ((yMax - v) / yRange) * plotH;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#181825';
    ctx.fillRect(padL, padT, plotW, plotH);

    // Grid — zero lines
    ctx.strokeStyle = '#45475a';
    ctx.lineWidth = 1;
    if (xMin < 0 && xMax > 0) {
      const zx = toX(0);
      ctx.beginPath(); ctx.moveTo(zx, padT); ctx.lineTo(zx, padT + plotH); ctx.stroke();
    }
    if (yMin < 0 && yMax > 0) {
      const zy = toY(0);
      ctx.beginPath(); ctx.moveTo(padL, zy); ctx.lineTo(padL + plotW, zy); ctx.stroke();
    }

    // Curve
    ctx.strokeStyle = '#f9e2af';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const xVal = xMin + (i / 200) * xRange;
      const yVal = curveFn(xVal);
      const cx = toX(xVal);
      const cy = toY(yVal);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Axis labels
    ctx.fillStyle = '#585b70';
    ctx.font = '11px JetBrains Mono, monospace';
    // X axis
    ctx.textAlign = 'center';
    ctx.fillText(fmtAxis(xMin), padL, padT + plotH + 16);
    ctx.fillText('0', toX(0), padT + plotH + 16);
    ctx.fillText(fmtAxis(xMax), padL + plotW, padT + plotH + 16);
    // Y axis
    ctx.textAlign = 'right';
    ctx.fillText(fmtAxis(yMax), padL - 5, padT + 10);
    ctx.fillText('0', padL - 5, toY(0) + 4);
    ctx.fillText(fmtAxis(yMin), padL - 5, padT + plotH + 4);
    // Axis titles
    ctx.fillStyle = '#6c7086';
    ctx.textAlign = 'center';
    ctx.fillText('input', padL + plotW / 2, H - 2);
    ctx.save();
    ctx.translate(10, padT + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('output', 0, 0);
    ctx.restore();
  }, [fn, xMin, xMax, negativeSlope]);

  return <canvas ref={canvasRef} className="tfm-activation-curve-canvas" />;
}

/** Histogram drawn within a fixed x-range (no curve overlay). */
function FixedRangeHistogram({ data, color, label, xMin, xMax, yMax }: {
  data: HistogramData; color: string; label: string;
  xMin: number; xMax: number; yMax: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.bins.length === 0) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    const padBot = 24;
    const plotH = h - padBot;
    const plotW = w;
    const xRange = xMax - xMin || 1;
    const binStep = data.bins.length > 1 ? data.bins[1] - data.bins[0] : 1;

    ctx.clearRect(0, 0, w, h);

    // Bars — tallest bar touches the top
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < data.counts.length; i++) {
      const binStart = data.bins[i];
      const x0 = ((binStart - xMin) / xRange) * plotW;
      const x1 = ((binStart + binStep - xMin) / xRange) * plotW;
      const barH = (data.counts[i] / yMax) * plotH;
      ctx.fillRect(x0, plotH - barH, Math.max(1, x1 - x0 - 0.5), barH);
    }
    ctx.globalAlpha = 1;

    // Zero line
    if (xMin < 0 && xMax > 0) {
      const zeroX = ((0 - xMin) / xRange) * plotW;
      ctx.strokeStyle = '#cdd6f4';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(zeroX, 0);
      ctx.lineTo(zeroX, plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Stats
    ctx.fillStyle = '#a6adc8';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`\u03BC=${fmtAxis(data.mean)}  \u03C3=${fmtAxis(data.std)}`, 2, h - 3);
    ctx.textAlign = 'right';
    ctx.fillText(`[${fmtAxis(xMin)}, ${fmtAxis(xMax)}]`, w - 2, h - 3);
  }, [data, color, xMin, xMax, yMax]);

  return (
    <div>
      <div className="tfm-panel-label">{label}</div>
      <canvas ref={canvasRef} className="tfm-activation-hist-canvas" />
    </div>
  );
}
