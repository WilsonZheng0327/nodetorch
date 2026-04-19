// Floating visualization panel that renders above a node.
// Shows weight distribution, gradient distribution, and activation histogram.
// Data comes from either live training snapshots or lastResult metadata.

import { useRef, useEffect, useState } from 'react';
import { useStore } from '@xyflow/react';
import './VizPanel.css';

interface HistogramData {
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  norm?: number;
  sparsity?: number;
  histBins?: number[];
  histCounts?: number[];
}

export interface VizSnapshot {
  weights?: HistogramData;
  gradients?: HistogramData;
  activations?: HistogramData;
  batchnorm?: {
    runningMean?: HistogramData;
    runningVar?: HistogramData;
  };
  weightDelta?: number;
}

interface Props {
  snapshot: VizSnapshot;
  onClose: () => void;
}

export function VizPanel({ snapshot, onClose }: Props) {
  const hasWeights = snapshot.weights?.histBins;
  const hasGradients = snapshot.gradients?.histBins;
  const hasActivations = snapshot.activations?.histBins;
  const hasBatchNorm = snapshot.batchnorm?.runningMean?.histBins;
  const hasWeightDelta = snapshot.weightDelta != null;
  const hasData = hasWeights || hasGradients || hasActivations || hasBatchNorm || hasWeightDelta;

  return (
    <div className="viz-panel nodrag">
      <div className="viz-panel-header">
        <span>Visualization</span>
        <button className="viz-panel-close" onClick={onClose}>&times;</button>
      </div>

      <div className="viz-panel-content">
        {!hasData ? (
          <div className="viz-panel-empty">No data yet — run forward or train</div>
        ) : (
          <>
            {hasWeights && (
              <VizSection title="Weights" color="#89b4fa" data={snapshot.weights!} tip="Distribution of all parameter values. Should be centered near 0. Collapse to 0 or explosion means trouble." stats={[
                { label: 'mean', value: snapshot.weights!.mean },
                { label: 'std', value: snapshot.weights!.std },
              ]} />
            )}

            {hasGradients && (
              <VizSection title="Gradients" color="#f9e2af" data={snapshot.gradients!} tip="Gradient magnitudes from backward pass. Tiny = vanishing gradients. Huge = exploding gradients." stats={[
                { label: 'mean', value: snapshot.gradients!.mean },
                { label: 'norm', value: snapshot.gradients!.norm },
              ]} warning={detectGradientIssue(snapshot.gradients!)} />
            )}

            {hasActivations && (
              <VizSection title="Activations" color="#10b981" data={snapshot.activations!} tip="Distribution of this layer's output values. High sparsity may mean dead neurons." stats={[
                { label: 'mean', value: snapshot.activations!.mean },
                { label: 'sparsity', value: snapshot.activations!.sparsity, pct: true },
              ]} warning={detectActivationIssue(snapshot.activations!)} />
            )}

            {hasBatchNorm && (
              <>
                <VizSection title="Running Mean" color="#cba6f7" data={snapshot.batchnorm!.runningMean!} tip="BatchNorm's learned per-channel mean of inputs." stats={[
                  { label: 'mean', value: snapshot.batchnorm!.runningMean!.mean },
                  { label: 'std', value: snapshot.batchnorm!.runningMean!.std },
                ]} />
                {snapshot.batchnorm!.runningVar?.histBins && (
                  <VizSection title="Running Var" color="#f5c2e7" data={snapshot.batchnorm!.runningVar!} tip="BatchNorm's learned per-channel variance of inputs." stats={[
                    { label: 'mean', value: snapshot.batchnorm!.runningVar!.mean },
                    { label: 'std', value: snapshot.batchnorm!.runningVar!.std },
                  ]} />
                )}
              </>
            )}

            {hasWeightDelta && (
              <WeightDeltaSection value={snapshot.weightDelta!} />
            )}
          </>
        )}
      </div>
    </div>
  );
}

// --- Health heuristics ---

function detectGradientIssue(g: HistogramData): string | null {
  if (g.norm != null) {
    if (g.norm < 1e-7) return 'Vanishing';
    if (g.norm > 100) return 'Exploding';
  }
  return null;
}

function detectActivationIssue(a: HistogramData): string | null {
  // Skip checks for scalar outputs (e.g., loss nodes)
  if (a.histCounts && a.histCounts.reduce((s, v) => s + v, 0) <= 1) return null;
  if (a.sparsity != null && a.sparsity > 0.9) return 'Dead neurons';
  if (a.std != null && a.std < 0.01) return 'Saturated';
  return null;
}

// --- Section with stats + histogram ---

function VizSection({ title, color, data, stats, tip, warning }: {
  title: string;
  color: string;
  data: HistogramData;
  stats: { label: string; value?: number; pct?: boolean }[];
  tip?: string;
  warning?: string | null;
}) {
  const [showTip, setShowTip] = useState(false);

  return (
    <div className="viz-section">
      <div className="viz-section-title" style={{ color }}>
        {title}
        {warning && <span className="viz-warning">{warning}</span>}
        {tip && <span className="viz-tip" onClick={() => setShowTip(!showTip)}>?</span>}
      </div>
      {showTip && tip && <div className="viz-tip-text">{tip}</div>}
      <div className="viz-stats">
        {stats.map((s) => (
          <span key={s.label}>
            {s.label}: {s.value != null
              ? (s.pct ? `${(s.value * 100).toFixed(1)}%` : s.value.toFixed(4))
              : '—'}
          </span>
        ))}
      </div>
      {data.histBins && data.histCounts && (
        <MiniHistogram bins={data.histBins} counts={data.histCounts} color={color} />
      )}
    </div>
  );
}

function WeightDeltaSection({ value }: { value: number }) {
  const [showTip, setShowTip] = useState(false);
  const warning = value < 1e-8 ? 'Not learning' : null;
  return (
    <div className="viz-section">
      <div className="viz-section-title" style={{ color: '#fab387' }}>
        Weight Delta
        {warning && <span className="viz-warning">{warning}</span>}
        <span className="viz-tip" onClick={() => setShowTip(!showTip)}>?</span>
      </div>
      {showTip && <div className="viz-tip-text">How much weights changed this epoch. Near-zero = converged or not learning.</div>}
      <div className="viz-stats">
        <span>norm: {value.toFixed(6)}</span>
      </div>
    </div>
  );
}

/** Format a number compactly for axis labels */
function formatAxis(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  if (abs >= 0.01) return v.toFixed(2);
  return v.toExponential(0);
}

// --- Canvas-rendered histogram ---

function MiniHistogram({ bins, counts, color }: { bins: number[]; counts: number[]; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Track canvas CSS size so the bitmap stays sharp when the panel resizes.
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  // React Flow zoom factor — the whole canvas area is CSS-scaled by this, so we
  // must render the bitmap at (layout_px × zoom × dpr) to stay crisp when zoomed in.
  const zoom = useStore((s) => s.transform[2]);

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
    if (!canvas || bins.length === 0 || size.w === 0) return;

    const dpr = window.devicePixelRatio || 1;
    // Upscale the bitmap by the current zoom (but never below 1x) so zooming in
    // on React Flow doesn't just stretch a low-res bitmap.
    const scale = dpr * Math.max(1, zoom);
    const w = size.w;
    const h = size.h;
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    const labelH = 14; // space for axis labels + ticks at bottom
    const tickH = 3;
    const plotH = h - labelH;
    const maxCount = Math.max(...counts);
    if (maxCount === 0) return;
    const barWidth = w / counts.length;

    const vmin = bins[0];
    const vmax = bins[bins.length - 1] + (bins.length > 1 ? bins[1] - bins[0] : 1);
    const vrange = vmax - vmin || 1;

    ctx.clearRect(0, 0, w, h);

    // Bars — align to device pixel boundaries to avoid anti-aliasing blur.
    // We draw on a DPR-scaled canvas; rounding in CSS pixels keeps edges crisp
    // because CSS pixels map to integer multiples of device pixels.
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < counts.length; i++) {
      const x = Math.round(i * barWidth);
      const nextX = Math.round((i + 1) * barWidth);
      const barH = Math.round((counts[i] / maxCount) * (plotH - 1));
      ctx.fillRect(x, plotH - barH, Math.max(1, nextX - x - 1), barH);
    }
    ctx.globalAlpha = 1;

    // X axis baseline
    ctx.strokeStyle = '#45475a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, plotH + 0.5);
    ctx.lineTo(w, plotH + 0.5);
    ctx.stroke();

    // X axis tick marks + labels (min, mid, max). Aligned so the min label sits
    // at the left edge and max at the right edge without clipping.
    const tickPositions: { frac: number; align: CanvasTextAlign; pxOffset: number }[] = [
      { frac: 0, align: 'left', pxOffset: 1 },
      { frac: 0.5, align: 'center', pxOffset: 0 },
      { frac: 1, align: 'right', pxOffset: -1 },
    ];
    ctx.strokeStyle = '#6c7086';
    ctx.fillStyle = '#a6adc8';
    ctx.font = '9px JetBrains Mono, monospace';
    for (const { frac, align, pxOffset } of tickPositions) {
      const xPos = frac * w;
      // tick mark
      ctx.beginPath();
      ctx.moveTo(Math.round(xPos) + 0.5, plotH);
      ctx.lineTo(Math.round(xPos) + 0.5, plotH + tickH);
      ctx.stroke();
      // label
      ctx.textAlign = align;
      const val = vmin + frac * vrange;
      ctx.fillText(formatAxis(val), xPos + pxOffset, h - 2);
    }

    // Zero marker (if 0 is within range and not already at an edge)
    if (vmin < 0 && vmax > 0) {
      const zeroX = ((0 - vmin) / vrange) * w;
      ctx.strokeStyle = '#cdd6f4';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(Math.round(zeroX) + 0.5, 0);
      ctx.lineTo(Math.round(zeroX) + 0.5, plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [bins, counts, color, size, zoom]);

  return (
    <canvas
      ref={canvasRef}
      className="viz-histogram"
    />
  );
}
