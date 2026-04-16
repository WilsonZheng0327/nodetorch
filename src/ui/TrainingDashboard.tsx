// Training dashboard — shows live training progress with loss/accuracy charts.

import { useRef, useEffect, useState } from 'react';
import './TrainingDashboard.css';

export interface EpochData {
  epoch: number;
  totalEpochs?: number;
  loss: number;
  accuracy: number;
  time?: number;
  batches?: number;
  samples?: number;
  gradientFlow?: { name: string; norm: number }[];
  perClassAccuracy?: { cls: number; accuracy: number }[];
}

interface Props {
  progress: EpochData[];
  isTraining: boolean;
}

export function TrainingDashboard({ progress, isTraining }: Props) {
  const [activeTab, setActiveTab] = useState<'loss' | 'accuracy' | 'gradients' | 'perclass'>('loss');
  const [open, setOpen] = useState(false);

  // Auto-open when training starts
  useEffect(() => {
    if (isTraining) setOpen(true);
  }, [isTraining]);

  // Nothing to show at all
  if (!isTraining && progress.length === 0) return null;

  const latest = progress.length > 0 ? progress[progress.length - 1] : null;
  const finished = !isTraining && progress.length > 0;

  if (!open) {
    return (
      <button
        className={`dashboard-toggle ${finished ? 'dashboard-toggle-done' : ''}`}
        onClick={() => setOpen(true)}
      >
        {latest
          ? `${finished ? 'Done' : 'Training'} — Epoch ${latest.epoch}, Loss ${latest.loss?.toFixed(4)}, Acc ${(latest.accuracy * 100).toFixed(1)}%`
          : 'Training — Starting...'}
      </button>
    );
  }

  return (
    <div className={`dashboard ${finished ? 'dashboard-done' : ''}`}>
      <div className="dashboard-header">
        <span className="dashboard-title">
          Training {finished && <span className="dashboard-done-badge">Complete</span>}
        </span>
        <button className="dashboard-close" onClick={() => setOpen(false)}>&ndash;</button>
      </div>

      {/* Progress bar */}
      {latest?.totalEpochs && (
        <div className="dashboard-progress">
          <div
            className="dashboard-progress-bar"
            style={{ width: `${(latest.epoch / latest.totalEpochs) * 100}%` }}
          />
          <span className="dashboard-progress-label">
            Epoch {latest.epoch} / {latest.totalEpochs}
          </span>
        </div>
      )}

      {/* Metrics summary */}
      <div className="dashboard-metrics">
        <div className="dashboard-metric">
          <span className="dashboard-metric-label">Loss</span>
          <span className="dashboard-metric-value">{latest ? latest.loss?.toFixed(4) : '—'}</span>
        </div>
        <div className="dashboard-metric">
          <span className="dashboard-metric-label">Accuracy</span>
          <span className="dashboard-metric-value">{latest ? `${(latest.accuracy * 100).toFixed(1)}%` : '—'}</span>
        </div>
        <div className="dashboard-metric">
          <span className="dashboard-metric-label">Time</span>
          <span className="dashboard-metric-value">{latest?.time != null ? `${latest.time}s` : '—'}</span>
        </div>
      </div>

      {/* Tab selector */}
      <div className="dashboard-tabs">
        <button
          className={`dashboard-tab ${activeTab === 'loss' ? 'dashboard-tab-active' : ''}`}
          onClick={() => setActiveTab('loss')}
        >
          Loss
        </button>
        <button
          className={`dashboard-tab ${activeTab === 'accuracy' ? 'dashboard-tab-active' : ''}`}
          onClick={() => setActiveTab('accuracy')}
        >
          Accuracy
        </button>
        <button
          className={`dashboard-tab ${activeTab === 'gradients' ? 'dashboard-tab-active' : ''}`}
          onClick={() => setActiveTab('gradients')}
        >
          Gradients
        </button>
        <button
          className={`dashboard-tab ${activeTab === 'perclass' ? 'dashboard-tab-active' : ''}`}
          onClick={() => setActiveTab('perclass')}
        >
          Per-Class
        </button>
      </div>

      {/* Chart */}
      {progress.length > 0 ? (
        activeTab === 'gradients' ? (
          <GradientFlowChart data={latest?.gradientFlow ?? []} />
        ) : activeTab === 'perclass' ? (
          <PerClassChart data={latest?.perClassAccuracy ?? []} />
        ) : (
          <Chart
            data={progress.map((d) => activeTab === 'loss' ? d.loss : d.accuracy)}
            labels={progress.map((d) => d.epoch)}
            color={activeTab === 'loss' ? '#ef4444' : '#10b981'}
            formatValue={activeTab === 'loss' ? (v) => v.toFixed(4) : (v) => (v * 100).toFixed(1) + '%'}
          />
        )
      ) : (
        <div className="dashboard-chart-placeholder">
          {isTraining ? 'Waiting for first epoch...' : 'No data yet'}
        </div>
      )}

      {/* Epoch table */}
      <div className="dashboard-table-wrap">
        <table className="dashboard-table">
          <thead>
            <tr>
              <th>Epoch</th>
              <th>Loss</th>
              <th>Accuracy</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {progress.map((d) => (
              <tr key={d.epoch}>
                <td>{d.epoch}</td>
                <td>{d.loss?.toFixed(4)}</td>
                <td>{(d.accuracy * 100).toFixed(1)}%</td>
                <td>{d.time != null ? `${d.time}s` : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// --- Canvas chart component ---

interface ChartProps {
  data: number[];
  labels: number[];
  color: string;
  formatValue: (v: number) => string;
}

function Chart({ data, labels, color, formatValue }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = { top: 20, right: 12, bottom: 24, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Clear
    ctx.clearRect(0, 0, w, h);

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    // Y axis labels
    ctx.fillStyle = '#6c7086';
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = min + (range * i) / 4;
      const y = pad.top + plotH - (plotH * i) / 4;
      ctx.fillText(formatValue(val), pad.left - 6, y + 3);

      // Grid line
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
  }, [data, labels, color, formatValue]);

  return <canvas ref={canvasRef} className="dashboard-chart" />;
}

// --- Gradient flow horizontal bar chart ---

function GradientFlowChart({ data }: { data: { name: string; norm: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const labelW = 80;
    const pad = { top: 8, right: 12, bottom: 8 };
    const plotW = w - labelW - pad.right;
    const barH = Math.min(20, (h - pad.top - pad.bottom) / data.length - 2);
    const maxNorm = Math.max(...data.map((d) => d.norm), 1e-8);

    ctx.clearRect(0, 0, w, h);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + 2);
      const barW = (data[i].norm / maxNorm) * plotW;

      // Label
      ctx.fillStyle = '#a6adc8';
      ctx.font = '10px Inter, system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(data[i].name, labelW - 6, y + barH / 2 + 3);

      // Bar background
      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      // Bar
      const ratio = data[i].norm / maxNorm;
      // Color: green→yellow→red based on relative magnitude
      const r = Math.round(Math.min(255, ratio * 2 * 255));
      const g = Math.round(Math.min(255, (1 - ratio) * 2 * 255));
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      // Value
      ctx.fillStyle = '#6c7086';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(data[i].norm.toExponential(1), labelW + barW + 4, y + barH / 2 + 3);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No gradient data yet</div>;
  }

  return <canvas ref={canvasRef} className="dashboard-chart" />;
}

// --- Per-class accuracy horizontal bar chart ---

function PerClassChart({ data }: { data: { cls: number; accuracy: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const labelW = 36;
    const pad = { top: 8, right: 50, bottom: 8 };
    const plotW = w - labelW - pad.right;
    const barH = Math.min(14, (h - pad.top - pad.bottom) / data.length - 1);

    ctx.clearRect(0, 0, w, h);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + 1);
      const barW = data[i].accuracy * plotW;

      // Class label
      ctx.fillStyle = '#6c7086';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(data[i].cls), labelW - 4, y + barH / 2 + 3);

      // Bar background
      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      // Bar — green for high accuracy, red for low
      const acc = data[i].accuracy;
      const r = Math.round((1 - acc) * 230);
      const g = Math.round(acc * 200);
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      // Value
      ctx.fillStyle = '#a6adc8';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${(acc * 100).toFixed(0)}%`, labelW + barW + 4, y + barH / 2 + 3);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No per-class data yet</div>;
  }

  return <canvas ref={canvasRef} className="dashboard-chart" />;
}
