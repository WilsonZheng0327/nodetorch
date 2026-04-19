// Training dashboard — always-visible panel with training progress, charts, and system info.

import { useRef, useEffect, useState } from 'react';
import './TrainingDashboard.css';

export interface EpochData {
  epoch: number;
  totalEpochs?: number;
  loss: number;
  accuracy: number;
  valLoss?: number | null;
  valAccuracy?: number | null;
  learningRate?: number | null;
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
  currentDevice?: string;
}

export interface ModelLayerInfo {
  name: string;
  type: string;
  paramCount?: number;
  outputShape?: number[] | string[];
}

export interface SavedRun {
  id: string;
  timestamp: string;
  datasetType: string;
  epochs: number;
  learningRate: number;
  optimizer: string;
  scheduler: string;
  finalLoss: number | null;
  finalAccuracy: number | null;
  bestValAccuracy: number | null;
  duration: number;
  totalParams: number;
  nodeCount: number;
}

export interface FullRun extends SavedRun {
  batchSize: number;
  seed: number;
  valSplit: number;
  epochHistory: EpochData[];
}

interface Props {
  progress: EpochData[];
  isTraining: boolean;
  batchProgress?: { batch: number; totalBatches: number } | null;
  selectedEpoch: number | null;
  onSelectEpoch: (epoch: number | null) => void;
  totalSnapshotEpochs: number;
  modelSummary?: ModelLayerInfo[];
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function TrainingDashboard({ progress, isTraining, batchProgress, selectedEpoch, onSelectEpoch, totalSnapshotEpochs, modelSummary, open: openProp, onOpenChange }: Props) {
  const [activeTab, setActiveTab] = useState<'loss' | 'accuracy' | 'gradients' | 'perclass' | 'epochs' | 'summary' | 'runs' | 'system'>('loss');
  const [savedRuns, setSavedRuns] = useState<SavedRun[] | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);
  const [compareRun, setCompareRun] = useState<FullRun | null>(null);
  const [internalOpen, setInternalOpen] = useState(false);
  // Support both controlled (via `open` prop) and uncontrolled use.
  const open = openProp ?? internalOpen;
  const setOpen = (v: boolean | ((prev: boolean) => boolean)) => {
    const next = typeof v === 'function' ? v(open) : v;
    if (onOpenChange) onOpenChange(next);
    else setInternalOpen(next);
  };
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

  // Fetch runs when tab becomes runs, and after training completes
  const fetchRuns = () => {
    setRunsLoading(true);
    fetch('http://localhost:8000/runs')
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setSavedRuns(data.runs);
      })
      .catch(() => {})
      .finally(() => setRunsLoading(false));
  };

  useEffect(() => {
    if (activeTab === 'runs' && savedRuns === null) fetchRuns();
  }, [activeTab, savedRuns]);

  // Refetch runs when training ends (a new run was just saved)
  const wasTraining = useRef(false);
  useEffect(() => {
    if (wasTraining.current && !isTraining) {
      setSavedRuns(null);  // force reload on next visit
    }
    wasTraining.current = isTraining;
  }, [isTraining]);

  const loadCompareRun = (id: string) => {
    fetch(`http://localhost:8000/runs/${id}`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setCompareRun(data.run);
      })
      .catch(() => {});
  };

  const deleteRun = (id: string) => {
    fetch(`http://localhost:8000/runs/${id}`, { method: 'DELETE' })
      .then(() => {
        setSavedRuns((prev) => prev ? prev.filter((r) => r.id !== id) : null);
        if (compareRun?.id === id) setCompareRun(null);
      })
      .catch(() => {});
  };

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
      {latest?.totalEpochs && (() => {
        const avgTime = progress.length > 0
          ? progress.reduce((s, d) => s + (d.time ?? 0), 0) / progress.length
          : 0;
        const remaining = (latest.totalEpochs - latest.epoch) * avgTime;
        const eta = remaining > 0 && isTraining
          ? remaining >= 60 ? `${Math.round(remaining / 60)}m ${Math.round(remaining % 60)}s` : `${Math.round(remaining)}s`
          : null;
        return (
          <div className="dashboard-progress">
            <div
              className="dashboard-progress-bar"
              style={{ width: `${(latest.epoch / latest.totalEpochs) * 100}%` }}
            />
            <span className="dashboard-progress-label">
              Epoch {latest.epoch} / {latest.totalEpochs}
              {eta && <span className="dashboard-eta"> — ~{eta} remaining</span>}
            </span>
          </div>
        );
      })()}

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
        {(['loss', 'accuracy', 'gradients', 'perclass', 'epochs', 'summary', 'runs', 'system'] as const).map((tab) => (
          <button
            key={tab}
            className={`dashboard-tab ${activeTab === tab ? 'dashboard-tab-active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {{ loss: 'Loss', accuracy: 'Accuracy', gradients: 'Gradients', perclass: 'Per-Class', epochs: 'Epochs', summary: 'Summary', runs: 'Runs', system: 'System' }[tab]}
          </button>
        ))}
      </div>

      {/* Epoch slider — scrub through training history */}
      {totalSnapshotEpochs >= 2 && activeTab !== 'system' && activeTab !== 'epochs' && activeTab !== 'summary' && activeTab !== 'runs' && (
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
        ) : activeTab === 'runs' ? (
          <RunsPanel
            runs={savedRuns}
            loading={runsLoading}
            onRefresh={fetchRuns}
            onCompare={loadCompareRun}
            onDelete={deleteRun}
            compareRun={compareRun}
            onClearCompare={() => setCompareRun(null)}
          />
        ) : activeTab === 'epochs' ? (
          <div className="dashboard-table-wrap">
            {progress.length > 0 ? (
              <table className="dashboard-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Loss</th>
                    <th>Acc</th>
                    <th>Val Loss</th>
                    <th>Val Acc</th>
                    <th>LR</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {progress.map((d) => (
                    <tr key={d.epoch}>
                      <td>{d.epoch}</td>
                      <td>{d.loss?.toFixed(4)}</td>
                      <td>{(d.accuracy * 100).toFixed(1)}%</td>
                      <td>{d.valLoss != null ? d.valLoss.toFixed(4) : '—'}</td>
                      <td>{d.valAccuracy != null ? `${(d.valAccuracy * 100).toFixed(1)}%` : '—'}</td>
                      <td>{d.learningRate != null ? d.learningRate.toExponential(1) : '—'}</td>
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
              valData={progress.map((d) => activeTab === 'loss' ? d.valLoss : d.valAccuracy)}
              compareData={compareRun ? compareRun.epochHistory.map((d) => activeTab === 'loss' ? d.loss : d.accuracy) : undefined}
              compareLabel={compareRun ? `${compareRun.optimizer} lr=${compareRun.learningRate} ep=${compareRun.epochs}` : undefined}
              labels={progress.map((d) => d.epoch)}
              color={activeTab === 'loss' ? '#ef4444' : '#10b981'}
              valColor="#fab387"
              compareColor="#cba6f7"
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
  const [currentDevice, setCurrentDevice] = useState<string>('cpu');

  useEffect(() => {
    if (info?.currentDevice) setCurrentDevice(info.currentDevice);
  }, [info]);

  if (!info) {
    return <div className="dashboard-chart-placeholder">Connecting to backend...</div>;
  }

  // Build available device options
  const deviceOptions: { value: string; label: string }[] = [
    { value: 'cpu', label: 'CPU' },
  ];
  if (info.cudaAvailable) {
    for (let i = 0; i < info.gpuCount; i++) {
      const name = info.gpus[i]?.name ?? `GPU ${i}`;
      deviceOptions.push({ value: i === 0 ? 'cuda' : `cuda:${i}`, label: `CUDA: ${name}` });
    }
  }
  if (info.mpsAvailable) {
    deviceOptions.push({ value: 'mps', label: 'MPS (Apple Silicon)' });
  }

  function handleDeviceChange(device: string) {
    setCurrentDevice(device);
    fetch('http://localhost:8000/set-device', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ device }),
    }).catch(() => {});
  }

  return (
    <div className="system-info">
      {/* Device selector */}
      <div className="system-info-device">
        <span className="system-info-label">Training Device</span>
        <select
          className="system-info-device-select"
          value={currentDevice}
          onChange={(e) => handleDeviceChange(e.target.value)}
        >
          {deviceOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

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
        <div className="system-info-note">No GPU available — training will use CPU</div>
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
  valData?: (number | null | undefined)[];
  valColor?: string;
  compareData?: (number | null | undefined)[];
  compareColor?: string;
  compareLabel?: string;
}

function Chart({ data: rawData, labels, color, formatValue, selectedIndex, valData, valColor, compareData, compareColor, compareLabel }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Track canvas client size so we re-render on resize (keeps bitmap crisp)
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

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
    if (!canvas || rawData.length === 0 || size.w === 0) return;

    // Guard against NaN/null/Infinity
    const data = rawData.map((v) => (v == null || !isFinite(v)) ? 0 : v);
    // Sanitize valData (preserve null = "no val for this epoch")
    const valClean = valData?.map((v) => (v != null && isFinite(v) ? v : null));
    const compareClean = compareData?.map((v) => (v != null && isFinite(v) ? v : null));

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    // Use clientWidth/Height (excludes padding) for accurate bitmap sizing.
    const w = size.w;
    const h = size.h;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { top: 40, right: 12, bottom: 28, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Compute min/max across all series
    const valValuesForRange = (valClean?.filter((v): v is number => v != null)) ?? [];
    const compareValuesForRange = (compareClean?.filter((v): v is number => v != null)) ?? [];
    const allValues = [...data, ...valValuesForRange, ...compareValuesForRange];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    // Y axis labels
    ctx.fillStyle = '#6c7086';
    ctx.font = '12px Inter, system-ui, sans-serif';
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

    // Validation line (dashed) if provided
    if (valClean) {
      const vColor = valColor ?? '#fab387';
      ctx.strokeStyle = vColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < valClean.length; i++) {
        const v = valClean[i];
        if (v == null) {
          started = false;
          continue;
        }
        const x = pad.left + (plotW * i) / Math.max(valClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // Val points
      ctx.fillStyle = vColor;
      for (let i = 0; i < valClean.length; i++) {
        const v = valClean[i];
        if (v == null) continue;
        const x = pad.left + (plotW * i) / Math.max(valClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Compare line (dotted, purple by default)
    if (compareClean && compareClean.length > 0) {
      const cColor = compareColor ?? '#cba6f7';
      ctx.strokeStyle = cColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([1, 3]);
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < compareClean.length; i++) {
        const v = compareClean[i];
        if (v == null) {
          started = false;
          continue;
        }
        const x = pad.left + (plotW * i) / Math.max(compareClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // Legend (if val line present). Sits in the top margin with breathing room above.
    if (valClean) {
      const legendY = 18;       // text baseline
      const legendRectY = 12;   // line sample top
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'left';
      // Train legend
      ctx.fillStyle = color;
      ctx.fillRect(pad.left + 4, legendRectY, 10, 2);
      ctx.fillStyle = '#a6adc8';
      ctx.fillText('train', pad.left + 18, legendY);
      // Val legend
      ctx.strokeStyle = valColor ?? '#fab387';
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(pad.left + 60, legendRectY + 1);
      ctx.lineTo(pad.left + 70, legendRectY + 1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#a6adc8';
      ctx.fillText('val', pad.left + 74, legendY);
      // Compare legend
      if (compareClean && compareClean.length > 0) {
        ctx.strokeStyle = compareColor ?? '#cba6f7';
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(pad.left + 110, legendRectY + 1);
        ctx.lineTo(pad.left + 120, legendRectY + 1);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#cba6f7';
        const labelText = compareLabel && compareLabel.length < 40 ? compareLabel : 'compare';
        ctx.fillText(labelText, pad.left + 124, legendY);
      }
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
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(formatValue(data[selectedIndex]), sx, sy - 10);
    }
  }, [rawData, labels, color, formatValue, selectedIndex, valData, valColor, compareData, compareColor, compareLabel, size]);

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
    const barH = 22;
    const gap = 2;
    const labelW = 160;
    const pad = { top: 8, right: 80, bottom: 8 };
    const totalH = pad.top + data.length * (barH + gap) + pad.bottom;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = totalH * dpr;
    canvas.style.height = `${totalH}px`;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const plotW = w - labelW - pad.right;
    const maxNorm = Math.max(...data.map((d) => d.norm), 1e-8);

    ctx.clearRect(0, 0, w, totalH);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + gap);
      const barW = (data[i].norm / maxNorm) * plotW;

      // Label (truncated if too long for labelW)
      ctx.fillStyle = '#a6adc8';
      ctx.font = '12px Inter, system-ui, sans-serif';
      ctx.textAlign = 'right';
      let label = data[i].name;
      if (ctx.measureText(label).width > labelW - 8) {
        while (label.length > 3 && ctx.measureText(label + '…').width > labelW - 8) {
          label = label.slice(0, -1);
        }
        label = label + '…';
      }
      ctx.fillText(label, labelW - 6, y + barH / 2 + 3);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const ratio = data[i].norm / maxNorm;
      const r = Math.round(Math.min(255, ratio * 2 * 255));
      const g = Math.round(Math.min(255, (1 - ratio) * 2 * 255));
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      ctx.fillStyle = '#6c7086';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(data[i].norm.toExponential(1), labelW + barW + 4, y + barH / 2 + 3);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No gradient data yet</div>;
  }

  return (
    <div className="dashboard-gradflow-scroll">
      <div className="dashboard-explainer">
        Gradient magnitude (L2 norm) per layer after backward pass. Bars near zero at early
        layers can indicate vanishing gradients; very large bars can indicate exploding
        gradients. Healthy training usually shows gradients within a few orders of magnitude.
      </div>
      <canvas ref={canvasRef} className="dashboard-chart dashboard-chart-tall" />
    </div>
  );
}

// --- Per-class accuracy bar chart (all classes, sorted worst-first, scrollable) ---

// --- Runs panel (training history) ---

function RunsPanel({
  runs, loading, onRefresh, onCompare, onDelete, compareRun, onClearCompare,
}: {
  runs: SavedRun[] | null;
  loading: boolean;
  onRefresh: () => void;
  onCompare: (id: string) => void;
  onDelete: (id: string) => void;
  compareRun: FullRun | null;
  onClearCompare: () => void;
}) {
  if (loading) {
    return <div className="dashboard-chart-placeholder">Loading runs...</div>;
  }
  if (!runs || runs.length === 0) {
    return <div className="dashboard-chart-placeholder">No saved runs yet — train a model to create one</div>;
  }
  return (
    <div className="runs-panel">
      <div className="runs-panel-header">
        <span>{runs.length} saved run{runs.length === 1 ? '' : 's'}</span>
        {compareRun && (
          <button className="dashboard-tab" onClick={onClearCompare} title="Stop comparing">
            Clear compare ({compareRun.id.slice(-8)})
          </button>
        )}
        <button className="dashboard-tab" onClick={onRefresh} style={{ marginLeft: 'auto' }}>Refresh</button>
      </div>
      <table className="dashboard-table">
        <thead>
          <tr>
            <th>When</th>
            <th>Dataset</th>
            <th>Opt</th>
            <th>LR</th>
            <th>Ep</th>
            <th>Params</th>
            <th>Val Acc</th>
            <th>Best Val</th>
            <th>Dur</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id} className={compareRun?.id === r.id ? 'runs-row-active' : ''}>
              <td>{r.timestamp?.replace('T', ' ') ?? '—'}</td>
              <td>{r.datasetType?.split('.').pop()}</td>
              <td>{r.optimizer}</td>
              <td>{r.learningRate?.toExponential(1)}</td>
              <td>{r.epochs}</td>
              <td>{r.totalParams?.toLocaleString()}</td>
              <td>{r.bestValAccuracy != null ? `${(r.bestValAccuracy * 100).toFixed(1)}%` : '—'}</td>
              <td>{r.finalAccuracy != null ? `${(r.finalAccuracy * 100).toFixed(1)}%` : '—'}</td>
              <td>{r.duration}s</td>
              <td>
                <button className="runs-action-btn" onClick={() => onCompare(r.id)} title="Overlay on loss/accuracy charts">
                  Compare
                </button>
                <button className="runs-action-btn runs-action-delete" onClick={() => onDelete(r.id)} title="Delete">
                  ×
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerClassChart({ data }: { data: { cls: number; accuracy: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const barH = 24;
    const gap = 3;
    const labelW = 48;
    const pad = { top: 8, right: 60, bottom: 8 };
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

      ctx.fillStyle = '#a6adc8';
      ctx.font = '13px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(data[i].cls), labelW - 6, y + barH / 2 + 4);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const acc = data[i].accuracy;
      const r = Math.round((1 - acc) * 230);
      const g = Math.round(acc * 200);
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      ctx.fillStyle = '#cdd6f4';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${(acc * 100).toFixed(1)}%`, labelW + barW + 6, y + barH / 2 + 4);
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
