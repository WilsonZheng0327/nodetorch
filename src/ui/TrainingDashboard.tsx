// Training dashboard — always-visible panel with training progress, charts, and system info.

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

interface SystemInfo {
  python: string;
  pytorch: string;
  cudaAvailable: boolean;
  gpuCount: number;
  gpus: { name: string; vram: number; computeCapability: string }[];
  mpsAvailable?: boolean;
}

interface Props {
  progress: EpochData[];
  isTraining: boolean;
}

export function TrainingDashboard({ progress, isTraining }: Props) {
  const [activeTab, setActiveTab] = useState<'loss' | 'accuracy' | 'gradients' | 'perclass' | 'epochs' | 'system'>('loss');
  const [open, setOpen] = useState(false);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  // Fetch system info on mount
  useEffect(() => {
    fetch('http://localhost:8000/system-info')
      .then((r) => r.json())
      .then((data) => setSystemInfo(data))
      .catch(() => {/* backend not running */});
  }, []);

  // Auto-open when training starts
  useEffect(() => {
    if (isTraining) setOpen(true);
  }, [isTraining]);

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
          : 'Dashboard'}
      </button>
    );
  }

  return (
    <div className={`dashboard ${finished ? 'dashboard-done' : ''}`}>
      <div className="dashboard-header">
        <span className="dashboard-title">
          Dashboard
          {isTraining && <span className="dashboard-training-badge">Training</span>}
          {finished && <span className="dashboard-done-badge">Complete</span>}
        </span>
        <button className="dashboard-close" onClick={() => setOpen(false)}>&ndash;</button>
      </div>

      {/* Progress bar — only during/after training */}
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
      {latest ? (
        <div className="dashboard-metrics">
          <div className="dashboard-metric">
            <span className="dashboard-metric-label">Loss</span>
            <span className="dashboard-metric-value">{latest.loss?.toFixed(4)}</span>
          </div>
          <div className="dashboard-metric">
            <span className="dashboard-metric-label">Accuracy</span>
            <span className="dashboard-metric-value">{(latest.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="dashboard-metric">
            <span className="dashboard-metric-label">Time</span>
            <span className="dashboard-metric-value">{latest.time != null ? `${latest.time}s` : '—'}</span>
          </div>
        </div>
      ) : (
        <div className="dashboard-metrics">
          <div className="dashboard-metric">
            <span className="dashboard-metric-label">Status</span>
            <span className="dashboard-metric-value" style={{ fontSize: 14 }}>No training data yet</span>
          </div>
        </div>
      )}

      {/* Tab selector */}
      <div className="dashboard-tabs">
        {(['loss', 'accuracy', 'gradients', 'perclass', 'epochs', 'system'] as const).map((tab) => (
          <button
            key={tab}
            className={`dashboard-tab ${activeTab === tab ? 'dashboard-tab-active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {{ loss: 'Loss', accuracy: 'Accuracy', gradients: 'Gradients', perclass: 'Per-Class', epochs: 'Epochs', system: 'System' }[tab]}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="dashboard-tab-content">
        {activeTab === 'system' ? (
          <SystemInfoPanel info={systemInfo} />
        ) : activeTab === 'epochs' ? (
          <div className="dashboard-table-wrap">
            {progress.length > 0 ? (
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
            ) : (
              <div className="dashboard-chart-placeholder">No epoch data yet — train to see results</div>
            )}
          </div>
        ) : progress.length > 0 ? (
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
            {isTraining ? 'Waiting for first epoch...' : 'No training data yet — train to see results'}
          </div>
        )}
      </div>
    </div>
  );
}

// --- System info panel ---

function SystemInfoPanel({ info }: { info: SystemInfo | null }) {
  if (!info) {
    return <div className="dashboard-chart-placeholder">Connecting to backend...</div>;
  }

  return (
    <div className="system-info">
      <div className="system-info-row">
        <span className="system-info-label">Python</span>
        <span className="system-info-value">{info.python}</span>
      </div>
      <div className="system-info-row">
        <span className="system-info-label">PyTorch</span>
        <span className="system-info-value">{info.pytorch}</span>
      </div>
      <div className="system-info-row">
        <span className="system-info-label">CUDA</span>
        <span className={`system-info-value ${info.cudaAvailable ? 'system-info-ok' : 'system-info-warn'}`}>
          {info.cudaAvailable ? 'Available' : 'Not available'}
        </span>
      </div>
      {info.mpsAvailable && (
        <div className="system-info-row">
          <span className="system-info-label">MPS (Apple)</span>
          <span className="system-info-value system-info-ok">Available</span>
        </div>
      )}
      {info.gpuCount > 0 && (
        <>
          <div className="system-info-row">
            <span className="system-info-label">GPU Count</span>
            <span className="system-info-value">{info.gpuCount}</span>
          </div>
          {info.gpus.map((gpu, i) => (
            <div key={i} className="system-info-gpu">
              <div className="system-info-gpu-name">{gpu.name}</div>
              <div className="system-info-gpu-detail">
                {gpu.vram} GB VRAM &middot; Compute {gpu.computeCapability}
              </div>
            </div>
          ))}
        </>
      )}
      {!info.cudaAvailable && !info.mpsAvailable && (
        <div className="system-info-note">Training will use CPU (slower)</div>
      )}
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

function Chart({ data: rawData, labels, color, formatValue }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rawData.length === 0) return;

    // Guard against NaN/null/Infinity
    const data = rawData.map((v) => (v == null || !isFinite(v)) ? 0 : v);

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
  }, [rawData, labels, color, formatValue]);

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
    const barH = Math.min(24, (h - pad.top - pad.bottom) / data.length - 2);
    const maxNorm = Math.max(...data.map((d) => d.norm), 1e-8);

    ctx.clearRect(0, 0, w, h);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + 2);
      const barW = (data[i].norm / maxNorm) * plotW;

      ctx.fillStyle = '#a6adc8';
      ctx.font = '10px Inter, system-ui, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(data[i].name, labelW - 6, y + barH / 2 + 3);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const ratio = data[i].norm / maxNorm;
      const r = Math.round(Math.min(255, ratio * 2 * 255));
      const g = Math.round(Math.min(255, (1 - ratio) * 2 * 255));
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

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

// --- Per-class accuracy bar chart (all classes, sorted worst-first, scrollable) ---

function PerClassChart({ data }: { data: { cls: number; accuracy: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const barH = 16;
    const gap = 1;
    const labelW = 36;
    const pad = { top: 8, right: 50, bottom: 8 };
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

      ctx.fillStyle = '#6c7086';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(data[i].cls), labelW - 4, y + barH / 2 + 3);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const acc = data[i].accuracy;
      const r = Math.round((1 - acc) * 230);
      const g = Math.round(acc * 200);
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      ctx.fillStyle = '#a6adc8';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${(acc * 100).toFixed(0)}%`, labelW + barW + 4, y + barH / 2 + 3);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No per-class data yet</div>;
  }

  return (
    <div className="dashboard-perclass-scroll">
      <canvas ref={canvasRef} className="dashboard-chart dashboard-chart-tall" />
    </div>
  );
}
