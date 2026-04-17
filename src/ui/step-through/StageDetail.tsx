// Full detail view of the currently-selected stage.
// Switches display based on viz.kind. Each kind is rendered by a small self-contained sub-component.

import { useRef, useEffect } from 'react';
import type { Stage, Stats } from './types';
import { formatShape } from './insights';

interface Props {
  stage: Stage;
}

export function StageDetail({ stage }: Props) {
  const { viz, stats, insight } = stage;

  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">{stage.displayName}</span>
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
  return (
    <div>
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

function StatsBar({ stats }: { stats: Stats }) {
  return (
    <div className="stage-detail-stats">
      {stats.mean != null && <span>mean: {stats.mean.toFixed(4)}</span>}
      {stats.std != null && <span>std: {stats.std.toFixed(4)}</span>}
      {stats.min != null && <span>min: {stats.min.toFixed(3)}</span>}
      {stats.max != null && <span>max: {stats.max.toFixed(3)}</span>}
      {stats.sparsity != null && <span>sparsity: {(stats.sparsity * 100).toFixed(0)}%</span>}
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
    const w = rect.width;
    const h = rect.height;
    const min = Math.min(0, ...values);
    const max = Math.max(0, ...values);
    const range = max - min || 1;
    ctx.clearRect(0, 0, w, h);

    const bw = w / values.length;
    // Zero line
    const zeroY = h - ((0 - min) / range) * h;
    for (let i = 0; i < values.length; i++) {
      const bh = Math.abs(values[i] / range) * h;
      const y = values[i] >= 0 ? zeroY - bh : zeroY;
      ctx.fillStyle = values[i] >= 0 ? '#89b4fa' : '#f38ba8';
      ctx.globalAlpha = 0.7;
      ctx.fillRect(i * bw, y, Math.max(1, bw - 0.5), bh);
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
  }, [values]);
  return <canvas ref={canvasRef} className="stage-detail-vector-chart" />;
}
