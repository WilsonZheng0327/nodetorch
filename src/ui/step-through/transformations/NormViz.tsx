// Normalization transformation: before/after distributions

import { useRef, useEffect } from 'react';
import type { NormTransformation, HistogramData } from '../types';
import { Arrow } from './shared';

export function NormViz({ t }: { t: NormTransformation }) {
  return (
    <div className="tfm-norm">
      <div className="tfm-flow">
        <div className="tfm-panel">
          <div className="tfm-panel-label">Before</div>
          <Histogram data={t.inputHist} color="#6c7086" />
        </div>

        <Arrow label={t.normKind} />

        <div className="tfm-panel">
          <div className="tfm-panel-label">After</div>
          <Histogram data={t.outputHist} color="#89b4fa" />
        </div>
      </div>

      {(t.gamma || t.beta) && (
        <div className="tfm-section">
          <div className="tfm-section-title">Learned Parameters</div>
          <div className="tfm-norm-params">
            {t.gamma && (
              <div className="tfm-norm-param">
                <span className="tfm-norm-param-label">&gamma; (scale)</span>
                <ParamBars values={t.gamma} />
              </div>
            )}
            {t.beta && (
              <div className="tfm-norm-param">
                <span className="tfm-norm-param-label">&beta; (shift)</span>
                <ParamBars values={t.beta} />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Histogram({ data, color }: { data: HistogramData; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.bins.length === 0 || data.counts.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const labelH = 16;
    const plotH = h - labelH;
    const maxCount = Math.max(...data.counts);
    if (maxCount === 0) return;
    const barWidth = w / data.counts.length;

    const vmin = data.bins[0];
    const vmax = data.bins[data.bins.length - 1] + (data.bins.length > 1 ? data.bins[1] - data.bins[0] : 1);
    const vrange = vmax - vmin || 1;

    ctx.clearRect(0, 0, w, h);

    // Bars
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.7;
    for (let i = 0; i < data.counts.length; i++) {
      const barH = (data.counts[i] / maxCount) * (plotH - 2);
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
    }

    // Stats label
    ctx.fillStyle = '#a6adc8';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`\u03BC=${fmt(data.mean)}  \u03C3=${fmt(data.std)}`, 2, h - 2);
  }, [data, color]);

  return <canvas ref={canvasRef} className="tfm-histogram-canvas" />;
}

function ParamBars({ values }: { values: number[] }) {
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
    const max = Math.max(...values.map(Math.abs), 0.01);
    const bw = W / values.length;
    ctx.clearRect(0, 0, W, H);
    const zeroY = H / 2;
    for (let i = 0; i < values.length; i++) {
      const bh = (Math.abs(values[i]) / max) * (H / 2 - 1);
      const y = values[i] >= 0 ? zeroY - bh : zeroY;
      ctx.fillStyle = values[i] >= 0 ? '#a6e3a1' : '#f9e2af';
      ctx.globalAlpha = 0.8;
      ctx.fillRect(i * bw, y, Math.max(1, bw - 0.5), bh);
    }
    ctx.globalAlpha = 1;
  }, [values]);

  return <canvas ref={canvasRef} className="tfm-param-bars-canvas" />;
}

function fmt(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}
