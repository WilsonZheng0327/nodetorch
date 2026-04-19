// Full detail view of the currently-selected stage.
// Switches display based on viz.kind. Each kind is rendered by a small self-contained sub-component.

import { useRef, useEffect } from 'react';
import type { Stage, Stats, ParamGradStats } from './types';
import { formatShape } from './insights';
import { ExtraPanels } from './ExtraPanels';

interface Props {
  stage: Stage;
}

export function StageDetail({ stage }: Props) {
  const { viz, stats, insight } = stage;

  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">
          {stage.blockName && (
            <span className="stage-detail-block">{stage.blockName}</span>
          )}
          {stage.displayName}
        </span>
        <span className="stage-detail-shape">
          {formatShape(stage.inputShape)} &rarr; {formatShape(stage.outputShape)}
        </span>
      </div>

      {insight && <div className="stage-detail-insight">{insight}</div>}

      <div className="stage-detail-body">
        {viz?.kind === 'feature_maps' && viz.featureMaps && (
          <FeatureMapsDetail data={viz.featureMaps} />
        )}
        {viz?.kind === 'vector' && viz.vector && (
          <VectorDetail data={viz.vector} />
        )}
        {viz?.kind === 'probabilities' && viz.probabilities && (
          <ProbabilitiesDetail data={viz.probabilities} />
        )}
        {viz?.kind === 'scalar' && viz.scalar && (
          <ScalarDetail value={viz.scalar.value} />
        )}
        {!viz && <div className="stage-detail-empty">No visualization available</div>}
      </div>

      {stats && <StatsBar stats={stats} />}

      {'paramGradStats' in stage && (stage as any).paramGradStats && (
        <ParamGradStatsBar stats={(stage as any).paramGradStats} />
      )}

      {stage.extras && stage.extras.length > 0 && <ExtraPanels extras={stage.extras} />}
    </div>
  );
}

// --- Sub-components per viz kind ---

function FeatureMapsDetail({ data }: { data: { maps: number[][][]; channels: number; showing: number; height: number; width: number } }) {
  return (
    <div>
      <div className="stage-detail-section-label">
        Channels {data.showing} of {data.channels} &middot; {data.height}×{data.width} each
      </div>
      <div className="stage-detail-fmap-grid">
        {data.maps.map((m, i) => (
          <FeatureMapCanvas key={i} pixels={m} label={`ch ${i}`} />
        ))}
      </div>
    </div>
  );
}

function VectorDetail({ data }: { data: { values: number[]; totalLength: number; truncated: boolean } }) {
  return (
    <div>
      <div className="stage-detail-section-label">
        Vector of {data.totalLength} values
        {data.truncated && <span> (showing first {data.values.length})</span>}
      </div>
      <VectorChart values={data.values} />
    </div>
  );
}

function ProbabilitiesDetail({ data }: { data: { values: number[]; topK: { index: number; value: number }[] } }) {
  const maxVal = Math.max(...data.values);
  const predicted = data.topK[0];
  return (
    <div>
      {predicted && (
        <div className="stage-detail-predicted">
          Predicted: <strong>class {predicted.index}</strong> ({(predicted.value * 100).toFixed(1)}%)
        </div>
      )}
      <div className="stage-detail-section-label">Top predictions</div>
      <div className="stage-detail-probs">
        {data.topK.map((p, i) => (
          <div key={i} className="stage-detail-prob-row">
            <span className="stage-detail-prob-idx">#{p.index}</span>
            <div className="stage-detail-prob-bar-bg">
              <div
                className="stage-detail-prob-bar"
                style={{ width: `${(p.value / maxVal) * 100}%` }}
              />
            </div>
            <span className="stage-detail-prob-val">{(p.value * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScalarDetail({ value }: { value: number }) {
  return (
    <div className="stage-detail-scalar">
      {value.toFixed(6)}
    </div>
  );
}

/** Format a stat value adaptively — use scientific notation for tiny values */
function fmtStat(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 100) return v.toFixed(1);
  if (abs >= 0.01) return v.toFixed(4);
  return v.toExponential(2);
}

function StatsBar({ stats }: { stats: Stats }) {
  return (
    <div className="stage-detail-stats">
      {stats.mean != null && <span>mean: {fmtStat(stats.mean)}</span>}
      {stats.std != null && <span>std: {fmtStat(stats.std)}</span>}
      {stats.min != null && <span>min: {fmtStat(stats.min)}</span>}
      {stats.max != null && <span>max: {fmtStat(stats.max)}</span>}
      {stats.sparsity != null && <span>sparsity: {(stats.sparsity * 100).toFixed(0)}%</span>}
    </div>
  );
}

function ParamGradStatsBar({ stats }: { stats: ParamGradStats }) {
  const healthColor = stats.health === 'healthy' ? '#10b981'
    : stats.health === 'vanishing' ? '#f9e2af'
    : '#ef4444';
  return (
    <div className="stage-detail-stats" style={{ borderLeft: `3px solid ${healthColor}` }}>
      <span>grad ‖∇‖: {fmtStat(stats.gradNorm)}</span>
      <span>weight ‖w‖: {fmtStat(stats.weightNorm)}</span>
      <span>ratio: {fmtStat(stats.ratio)}</span>
      <span style={{ color: healthColor, fontWeight: 600 }}>{stats.health}</span>
    </div>
  );
}

// --- Canvas helpers ---

function FeatureMapCanvas({ pixels, label }: { pixels: number[][]; label: string }) {
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
    <div className="stage-detail-fmap-item">
      <canvas ref={canvasRef} className="stage-detail-fmap-canvas" />
      <span className="stage-detail-fmap-label">{label}</span>
    </div>
  );
}

/** Format a number compactly for axis labels */
function fmtAxis(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  if (abs >= 0.01) return v.toFixed(2);
  return v.toExponential(0);
}

function VectorChart({ values }: { values: number[] }) {
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

    const yAxisW = 48;   // space for Y-axis labels
    const padTop = 10;   // so top label isn't clipped
    const padBot = 10;   // so bottom label isn't clipped
    const plotW = W - yAxisW;
    const plotH = H - padTop - padBot;

    const vmin = Math.min(0, ...values);
    const vmax = Math.max(0, ...values);
    const range = vmax - vmin || 1;
    ctx.clearRect(0, 0, W, H);

    // Y-axis gridlines and labels
    const nTicks = 4;
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'right';
    for (let t = 0; t <= nTicks; t++) {
      const frac = t / nTicks;
      const val = vmax - frac * range;
      const y = padTop + Math.round(frac * plotH);
      // gridline
      ctx.strokeStyle = '#313244';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(yAxisW, y + 0.5);
      ctx.lineTo(W, y + 0.5);
      ctx.stroke();
      // label
      ctx.fillStyle = '#6c7086';
      ctx.fillText(fmtAxis(val), yAxisW - 4, y + 4);
    }

    // Bars
    const bw = plotW / values.length;
    const zeroY = padTop + plotH - ((0 - vmin) / range) * plotH;
    for (let i = 0; i < values.length; i++) {
      const x = yAxisW + i * bw;
      const bh = Math.abs(values[i] / range) * plotH;
      const y = values[i] >= 0 ? zeroY - bh : zeroY;
      ctx.fillStyle = values[i] >= 0 ? '#89b4fa' : '#f38ba8';
      ctx.globalAlpha = 0.7;
      ctx.fillRect(x, y, Math.max(1, bw - 0.5), bh);
    }
    ctx.globalAlpha = 1;

    // Zero line
    if (vmin < 0 && vmax > 0) {
      ctx.strokeStyle = '#a6adc8';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(yAxisW, zeroY);
      ctx.lineTo(W, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [values]);
  return <canvas ref={canvasRef} className="stage-detail-vector-chart" />;
}
