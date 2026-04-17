// Small card shown in the horizontal timeline, one per stage.
// Shows layer name, output shape, and a tiny preview viz.

import { useRef, useEffect } from 'react';
import type { Stage } from './types';
import { compactShape } from './insights';

interface Props {
  stage: Stage;
  active: boolean;
  onClick: () => void;
}

export function StageCard({ stage, active, onClick }: Props) {
  return (
    <button
      className={`stage-card ${active ? 'stage-card-active' : ''}`}
      onClick={onClick}
      title={stage.displayName}
    >
      <div className="stage-card-name">{stage.displayName}</div>
      <div className="stage-card-preview">
        <StageMiniViz stage={stage} />
      </div>
      <div className="stage-card-shape">{compactShape(stage.outputShape)}</div>
    </button>
  );
}

// --- Small per-kind preview ---

function StageMiniViz({ stage }: { stage: Stage }) {
  const viz = stage.viz;
  if (!viz) return <div className="stage-card-empty">—</div>;

  if (viz.kind === 'feature_maps' && viz.featureMaps && viz.featureMaps.maps[0]) {
    return <MiniFeatureMap pixels={viz.featureMaps.maps[0]} />;
  }

  if (viz.kind === 'vector' && viz.vector) {
    return <MiniSparkline values={viz.vector.values} />;
  }

  if (viz.kind === 'probabilities' && viz.probabilities) {
    return <MiniSparkline values={viz.probabilities.values} color="#10b981" />;
  }

  if (viz.kind === 'scalar' && viz.scalar) {
    return <div className="stage-card-scalar">{viz.scalar.value.toFixed(3)}</div>;
  }

  return <div className="stage-card-empty">—</div>;
}

// --- Tiny canvas helpers ---

function MiniFeatureMap({ pixels }: { pixels: number[][] }) {
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
  return <canvas ref={canvasRef} className="stage-card-canvas" />;
}

function MiniSparkline({ values, color = '#89b4fa' }: { values: number[]; color?: string }) {
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
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    const bw = w / values.length;
    for (let i = 0; i < values.length; i++) {
      const bh = ((values[i] - min) / range) * (h - 1);
      ctx.fillRect(i * bw, h - bh, Math.max(1, bw - 0.5), bh);
    }
    ctx.globalAlpha = 1;
  }, [values, color]);
  return <canvas ref={canvasRef} className="stage-card-canvas" />;
}
