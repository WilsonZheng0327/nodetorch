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

export interface ModelLayerInfo {
  name: string;
  type: string;
  paramCount?: number;
  outputShape?: number[] | string[];
}

interface Props {
  progress: EpochData[];
  isTraining: boolean;
  batchProgress?: { batch: number; totalBatches: number } | null;
  selectedEpoch: number | null;
  onSelectEpoch: (epoch: number | null) => void;
  totalSnapshotEpochs: number;
  modelSummary?: ModelLayerInfo[];
}

export function TrainingDashboard({ progress, isTraining, batchProgress, selectedEpoch, onSelectEpoch, totalSnapshotEpochs, modelSummary }: Props) {
  const [activeTab, setActiveTab] = useState<'loss' | 'accuracy' | 'gradients' | 'perclass' | 'epochs' | 'summary' | 'system'>('loss');
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
  // The epoch data currently being viewed (selected via slider, or latest)
  const viewed = selectedEpoch != null && selectedEpoch <= progress.length
    ? progress[selectedEpoch - 1]
    : latest;

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

      {/* Batch progress bar (within epoch) */}
      {isTraining && batchProgress && (
        <div className="dashboard-batch-progress">
          <div
            className="dashboard-batch-progress-bar"
            style={{ width: `${(batchProgress.batch / batchProgress.totalBatches) * 100}%` }}
          />
          <span className="dashboard-batch-progress-label">
            Batch {batchProgress.batch} / {batchProgress.totalBatches}
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
        {(['loss', 'accuracy', 'gradients', 'perclass', 'epochs', 'summary', 'system'] as const).map((tab) => (
          <button
            key={tab}
            className={`dashboard-tab ${activeTab === tab ? 'dashboard-tab-active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {{ loss: 'Loss', accuracy: 'Accuracy', gradients: 'Gradients', perclass: 'Per-Class', epochs: 'Epochs', summary: 'Summary', system: 'System' }[tab]}
          </button>
        ))}
      </div>

      {/* Epoch slider — scrub through training history */}
      {totalSnapshotEpochs >= 2 && activeTab !== 'system' && activeTab !== 'epochs' && activeTab !== 'summary' && (
        <div className="dashboard-epoch-slider">
          <span className="dashboard-epoch-slider-label">Epoch</span>
          <input
            type="range"
            min={1}
            max={totalSnapshotEpochs}
            value={selectedEpoch ?? totalSnapshotEpochs}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              onSelectEpoch(v === totalSnapshotEpochs ? null : v);
            }}
            className="dashboard-epoch-slider-input"
          />
          <span className="dashboard-epoch-slider-value">
            {selectedEpoch ?? totalSnapshotEpochs} / {totalSnapshotEpochs}
          </span>
        </div>
      )}

      {/* Tab content */}
      <div className="dashboard-tab-content">
        {activeTab === 'system' ? (
          <SystemInfoPanel info={systemInfo} />
        ) : activeTab === 'summary' ? (
          <ModelSummaryPanel layers={modelSummary ?? []} />
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
            <GradientFlowChart data={viewed?.gradientFlow ?? []} />
          ) : activeTab === 'perclass' ? (
            <PerClassChart data={viewed?.perClassAccuracy ?? []} />
          ) : (
            <Chart
              data={progress.map((d) => activeTab === 'loss' ? d.loss : d.accuracy)}
              labels={progress.map((d) => d.epoch)}
              color={activeTab === 'loss' ? '#ef4444' : '#10b981'}
              formatValue={activeTab === 'loss' ? (v) => v.toFixed(4) : (v) => (v * 100).toFixed(1) + '%'}
              selectedIndex={selectedEpoch != null ? selectedEpoch - 1 : null}
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

// --- Model summary panel ---

function ModelSummaryPanel({ layers }: { layers: ModelLayerInfo[] }) {
  if (layers.length === 0) {
    return <div className="dashboard-chart-placeholder">Run forward pass to see model summary</div>;
  }

  const totalParams = layers.reduce((sum, l) => sum + (l.paramCount ?? 0), 0);

  return (
    <div className="model-summary">
      <div className="model-summary-total">
        <span>Total Parameters</span>
        <span className="model-summary-total-value">{totalParams.toLocaleString()}</span>
      </div>
      <table className="dashboard-table">
        <thead>
          <tr>
            <th>Layer</th>
            <th>Type</th>
            <th>Output</th>
            <th>Params</th>
          </tr>
        </thead>
        <tbody>
          {layers.map((l, i) => (
            <tr key={i}>
              <td style={{ color: '#cdd6f4' }}>{l.name}</td>
              <td>{l.type}</td>
              <td>{l.outputShape ? `[${l.outputShape.join(', ')}]` : '—'}</td>
              <td>{l.paramCount ? l.paramCount.toLocaleString() : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// --- Canvas chart component ---

interface ChartProps {
  data: number[];
  labels: number[];
  color: string;
  formatValue: (v: number) => string;
  selectedIndex?: number | null;
}

function Chart({ data: rawData, labels, color, formatValue, selectedIndex }: ChartProps) {
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

    // Selected epoch marker (vertical line + larger dot)
    if (selectedIndex != null && selectedIndex >= 0 && selectedIndex < data.length) {
      const sx = pad.left + (plotW * selectedIndex) / Math.max(data.length - 1, 1);
      const sy = pad.top + plotH - (plotH * (data[selectedIndex] - min)) / range;
      // Vertical line
      ctx.strokeStyle = '#89b4fa';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx, pad.top);
      ctx.lineTo(sx, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      // Highlight dot
      ctx.fillStyle = '#89b4fa';
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fill();
      // Value label
      ctx.fillStyle = '#cdd6f4';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(formatValue(data[selectedIndex]), sx, sy - 10);
    }
  }, [rawData, labels, color, formatValue, selectedIndex]);

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
