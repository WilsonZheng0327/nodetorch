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
}

interface Props {
  progress: EpochData[];
  isTraining: boolean;
}

export function TrainingDashboard({ progress, isTraining }: Props) {
  const [activeTab, setActiveTab] = useState<'loss' | 'accuracy'>('loss');
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
      </div>

      {/* Chart */}
      {progress.length > 0 ? (
        <Chart
          data={progress.map((d) => activeTab === 'loss' ? d.loss : d.accuracy)}
          labels={progress.map((d) => d.epoch)}
          color={activeTab === 'loss' ? '#ef4444' : '#10b981'}
          formatValue={activeTab === 'loss' ? (v) => v.toFixed(4) : (v) => (v * 100).toFixed(1) + '%'}
        />
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
