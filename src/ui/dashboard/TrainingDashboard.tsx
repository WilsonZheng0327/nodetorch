// Training dashboard — always-visible panel with training progress, charts, and
// system info. The individual panels/charts/sample views live in sibling
// modules (charts.tsx, panels.tsx, samples.tsx); this file is just the container:
// state, data fetching, the tab bar, the epoch slider, and tab routing.

import { useRef, useEffect, useState, Fragment } from 'react';
import './TrainingDashboard.css';
import { apiUrl } from '../../api/base';
import type { EpochData, SystemInfo, ModelLayerInfo, SavedRun, FullRun, TestResult } from './types';
import { Chart, GradientFlowChart, PerClassChart } from './charts';
import { SystemInfoPanel, ModelSummaryPanel, RunsPanel, TestResultView } from './panels';
import { GeneratedTextView, TrackedSamplesView } from './samples';

// Re-export the types external consumers (e.g. App.tsx) import from the
// dashboard entry point, so those import sites don't need to know about ./types.
export type { EpochData, ModelLayerInfo } from './types';

interface Props {
  progress: EpochData[];
  isTraining: boolean;
  batchProgress?: { batch: number; totalBatches: number } | null;
  selectedEpoch: number | null;
  onSelectEpoch: (epoch: number | null) => void;
  totalSnapshotEpochs: number;
  modelSummary?: ModelLayerInfo[];
  testResult?: TestResult | null;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function TrainingDashboard({ progress, isTraining, batchProgress, selectedEpoch, onSelectEpoch, totalSnapshotEpochs, modelSummary, testResult, open: openProp, onOpenChange }: Props) {
  const [activeTab, setActiveTab] = useState<'loss' | 'accuracy' | 'gradients' | 'perclass' | 'samples' | 'test' | 'epochs' | 'summary' | 'runs' | 'system'>('loss');
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
    fetch(apiUrl('/system-info'))
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
    fetch(apiUrl('/runs'))
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
    fetch(apiUrl(`/runs/${id}`))
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') setCompareRun(data.run);
      })
      .catch(() => {});
  };

  const deleteRun = (id: string) => {
    fetch(apiUrl(`/runs/${id}`), { method: 'DELETE' })
      .then(() => {
        setSavedRuns((prev) => prev ? prev.filter((r) => r.id !== id) : null);
        if (compareRun?.id === id) setCompareRun(null);
      })
      .catch(() => {});
  };

  const latest = progress.length > 0 ? progress[progress.length - 1] : null;
  const isAutoregressive = latest?.trainingMode === 'autoregressive';
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
              {eta && <span className="dashboard-eta">{' '}(~{eta} remaining)</span>}
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
          {isAutoregressive ? (
            <div className="dashboard-metric">
              <span className="dashboard-metric-label">Perplexity</span>
              <span className="dashboard-metric-value">{latest.perplexity?.toFixed(1) ?? '—'}</span>
            </div>
          ) : (
            <div className="dashboard-metric">
              <span className="dashboard-metric-label">Accuracy</span>
              <span className="dashboard-metric-value">{(latest.accuracy * 100).toFixed(1)}%</span>
            </div>
          )}
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

      {/* Tab selector — dividers after Samples, Model, and Runs */}
      <div className="dashboard-tabs">
        {(['loss', 'accuracy', 'gradients', 'perclass', 'samples', 'test', 'epochs'] as const)
          .filter((tab) => {
            if (isAutoregressive && (tab === 'perclass' || tab === 'test')) return false;
            return true;
          })
          .map((tab) => (
          <Fragment key={tab}>
            <button
              className={`dashboard-tab ${activeTab === tab ? 'dashboard-tab-active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {{ loss: 'Loss', accuracy: isAutoregressive ? 'Perplexity' : 'Accuracy', gradients: 'Gradients', perclass: 'Per-Class', samples: isAutoregressive ? 'Generated' : 'Samples', test: 'Test', epochs: 'Epochs' }[tab]}
            </button>
            {tab === 'samples' && <span className="dashboard-tab-divider" />}
          </Fragment>
        ))}
        {(['summary', 'runs', 'system'] as const).map((tab) => (
          <Fragment key={tab}>
            <button
              className={`dashboard-tab ${activeTab === tab ? 'dashboard-tab-active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {{ summary: 'Model', runs: 'Runs', system: 'System' }[tab]}
            </button>
            {tab === 'summary' && <span className="dashboard-tab-divider" />}
          </Fragment>
        ))}
      </div>

      {/* Epoch slider — scrub through training history */}
      {totalSnapshotEpochs >= 2 && activeTab !== 'system' && activeTab !== 'epochs' && activeTab !== 'summary' && activeTab !== 'runs' && activeTab !== 'test' && (
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
        ) : activeTab === 'samples' ? (
          isAutoregressive ? <GeneratedTextView progress={progress} /> : <TrackedSamplesView progress={progress} selectedEpoch={selectedEpoch} />
        ) : activeTab === 'test' ? (
          <TestResultView result={testResult} />
        ) : activeTab === 'epochs' ? (
          <div className="dashboard-table-wrap">
            {progress.length > 0 ? (
              <table className="dashboard-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Loss</th>
                    <th>{isAutoregressive ? 'PPL' : 'Acc'}</th>
                    <th>Val Loss</th>
                    <th>{isAutoregressive ? 'Val PPL' : 'Val Acc'}</th>
                    <th>LR</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {progress.map((d) => (
                    <tr key={d.epoch}>
                      <td>{d.epoch}</td>
                      <td>{d.loss?.toFixed(4)}</td>
                      <td>{isAutoregressive ? (d.perplexity?.toFixed(1) ?? '—') : `${(d.accuracy * 100).toFixed(1)}%`}</td>
                      <td>{d.valLoss != null ? d.valLoss.toFixed(4) : '—'}</td>
                      <td>{isAutoregressive ? (d.valPerplexity != null ? d.valPerplexity.toFixed(1) : '—') : (d.valAccuracy != null ? `${(d.valAccuracy * 100).toFixed(1)}%` : '—')}</td>
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
              data={progress.map((d) => activeTab === 'loss' ? d.loss : isAutoregressive ? (d.perplexity ?? 0) : d.accuracy)}
              valData={progress.map((d) => activeTab === 'loss' ? d.valLoss : isAutoregressive ? (d.valPerplexity ?? null) : d.valAccuracy)}
              compareData={compareRun ? compareRun.epochHistory.map((d) => activeTab === 'loss' ? d.loss : d.accuracy) : undefined}
              compareLabel={compareRun ? `${compareRun.optimizer} lr=${compareRun.learningRate} ep=${compareRun.epochs}` : undefined}
              labels={progress.map((d) => d.epoch)}
              color={activeTab === 'loss' ? '#ef4444' : '#10b981'}
              valColor="#fab387"
              compareColor="#cba6f7"
              formatValue={activeTab === 'loss' ? (v) => v.toFixed(4) : isAutoregressive ? (v) => v.toFixed(1) : (v) => (v * 100).toFixed(1) + '%'}
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
