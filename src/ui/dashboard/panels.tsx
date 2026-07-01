// Non-chart dashboard panels: system info, model summary, saved-runs history,
// and the held-out test result. Styles come from TrainingDashboard.css.

import { useState } from 'react';
import { apiUrl } from '../../api/base';
import type { SystemInfo, ModelLayerInfo, SavedRun, FullRun, TestResult } from './types';

// --- System info panel ---

export function SystemInfoPanel({ info }: { info: SystemInfo | null }) {
  const [currentDevice, setCurrentDevice] = useState<string>('cpu');

  // Sync from the backend-reported device when it changes, via React's
  // adjust-state-during-render pattern (same as LeftRail) rather than an effect —
  // avoids a post-paint re-render. The selector can then override locally; a
  // later info refresh re-syncs only if the reported device actually differs.
  const [lastReported, setLastReported] = useState<string | undefined>(undefined);
  if (info?.currentDevice && info.currentDevice !== lastReported) {
    setLastReported(info.currentDevice);
    setCurrentDevice(info.currentDevice);
  }

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
    fetch(apiUrl('/set-device'), {
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

export function ModelSummaryPanel({ layers }: { layers: ModelLayerInfo[] }) {
  if (layers.length === 0) {
    return <div className="dashboard-chart-placeholder">Add nodes to see model parameters</div>;
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

// --- Runs panel (training history) ---

export function RunsPanel({
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

// --- Test Result View ---

export function TestResultView({ result }: { result?: TestResult | null }) {
  if (!result) {
    return <div className="dashboard-chart-placeholder">No test results yet — click "Test" after training to evaluate on the held-out test set</div>;
  }

  return (
    <div className="test-result-view">
      <div className="test-result-summary">
        <div className="test-result-metric">
          <span className="test-result-metric-value">{(result.testAccuracy * 100).toFixed(1)}%</span>
          <span className="test-result-metric-label">Test Accuracy</span>
        </div>
        <div className="test-result-metric">
          <span className="test-result-metric-value">{result.testLoss.toFixed(4)}</span>
          <span className="test-result-metric-label">Test Loss</span>
        </div>
        <div className="test-result-metric">
          <span className="test-result-metric-value">{result.testSamples.toLocaleString()}</span>
          <span className="test-result-metric-label">Samples</span>
        </div>
      </div>

      <div className="test-result-note">
        These results are on the <strong>held-out test set</strong> — data the model never saw during training.
        This measures how well the model generalizes to new, unseen data.
      </div>

      {result.perClassAccuracy.length > 0 && (
        <div className="test-result-perclass">
          <div className="test-result-section-title">Per-Class Accuracy</div>
          <div className="test-result-class-list">
            {result.perClassAccuracy.map((c) => (
              <div key={c.cls} className="test-result-class-row">
                <span className="test-result-class-name">{c.name}</span>
                <div className="test-result-class-bar-bg">
                  <div
                    className="test-result-class-bar"
                    style={{ width: `${c.accuracy * 100}%`, background: c.accuracy >= 0.8 ? '#10b981' : c.accuracy >= 0.5 ? '#f59e0b' : '#ef4444' }}
                  />
                </div>
                <span className="test-result-class-val">{(c.accuracy * 100).toFixed(1)}%</span>
                <span className="test-result-class-count">({c.count})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
