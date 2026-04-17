// Floating visualization panel that renders above a node.
// Shows weight distribution, gradient distribution, and activation histogram.
// Data comes from either live training snapshots or lastResult metadata.

import { useRef, useEffect } from 'react';
import './VizPanel.css';

interface HistogramData {
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  norm?: number;
  sparsity?: number;
  histBins?: number[];
  histCounts?: number[];
}

export interface VizSnapshot {
  weights?: HistogramData;
  gradients?: HistogramData;
  activations?: HistogramData;
  batchnorm?: {
    runningMean?: HistogramData;
    runningVar?: HistogramData;
  };
  weightDelta?: number;
}

interface Props {
  snapshot: VizSnapshot;
  onClose: () => void;
}

export function VizPanel({ snapshot, onClose }: Props) {
  const hasWeights = snapshot.weights?.histBins;
  const hasGradients = snapshot.gradients?.histBins;
  const hasActivations = snapshot.activations?.histBins;
  const hasBatchNorm = snapshot.batchnorm?.runningMean?.histBins;
  const hasWeightDelta = snapshot.weightDelta != null;
  const hasData = hasWeights || hasGradients || hasActivations || hasBatchNorm || hasWeightDelta;

  return (
    <div className="viz-panel nodrag nowheel">
      <div className="viz-panel-header">
        <span>Visualization</span>
        <button className="viz-panel-close" onClick={onClose}>&times;</button>
      </div>

      <div className="viz-panel-content">
        {!hasData ? (
          <div className="viz-panel-empty">No data yet — run forward or train</div>
        ) : (
          <>
            {hasWeights && (
              <VizSection title="Weights" color="#89b4fa" data={snapshot.weights!} tip="Distribution of all parameter values. Should be centered near 0. Collapse to 0 or explosion means trouble." stats={[
                { label: 'mean', value: snapshot.weights!.mean },
                { label: 'std', value: snapshot.weights!.std },
              ]} />
            )}

            {hasGradients && (
              <VizSection title="Gradients" color="#f9e2af" data={snapshot.gradients!} tip="Gradient magnitudes from backward pass. Tiny = vanishing gradients. Huge = exploding gradients." stats={[
                { label: 'mean', value: snapshot.gradients!.mean },
                { label: 'norm', value: snapshot.gradients!.norm },
              ]} />
            )}

            {hasActivations && (
              <VizSection title="Activations" color="#10b981" data={snapshot.activations!} tip="Distribution of this layer's output values. High sparsity may mean dead neurons." stats={[
                { label: 'mean', value: snapshot.activations!.mean },
                { label: 'sparsity', value: snapshot.activations!.sparsity, pct: true },
              ]} />
            )}

            {hasBatchNorm && (
              <>
                <VizSection title="Running Mean" color="#cba6f7" data={snapshot.batchnorm!.runningMean!} tip="BatchNorm's learned per-channel mean of inputs." stats={[
                  { label: 'mean', value: snapshot.batchnorm!.runningMean!.mean },
                  { label: 'std', value: snapshot.batchnorm!.runningMean!.std },
                ]} />
                {snapshot.batchnorm!.runningVar?.histBins && (
                  <VizSection title="Running Var" color="#f5c2e7" data={snapshot.batchnorm!.runningVar!} tip="BatchNorm's learned per-channel variance of inputs." stats={[
                    { label: 'mean', value: snapshot.batchnorm!.runningVar!.mean },
                    { label: 'std', value: snapshot.batchnorm!.runningVar!.std },
                  ]} />
                )}
              </>
            )}

            {hasWeightDelta && (
              <div className="viz-section">
                <div className="viz-section-title" style={{ color: '#fab387' }}>
                  Weight Delta
                  <span className="viz-tip" title="How much weights changed this epoch. Near-zero = converged or not learning.">?</span>
                </div>
                <div className="viz-stats">
                  <span>norm: {snapshot.weightDelta!.toFixed(6)}</span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// --- Section with stats + histogram ---

function VizSection({ title, color, data, stats, tip }: {
  title: string;
  color: string;
  data: HistogramData;
  stats: { label: string; value?: number; pct?: boolean }[];
  tip?: string;
}) {
  return (
    <div className="viz-section">
      <div className="viz-section-title" style={{ color }}>
        {title}
        {tip && <span className="viz-tip" title={tip}>?</span>}
      </div>
      <div className="viz-stats">
        {stats.map((s) => (
          <span key={s.label}>
            {s.label}: {s.value != null
              ? (s.pct ? `${(s.value * 100).toFixed(1)}%` : s.value.toFixed(4))
              : '—'}
          </span>
        ))}
      </div>
      {data.histBins && data.histCounts && (
        <MiniHistogram bins={data.histBins} counts={data.histCounts} color={color} />
      )}
    </div>
  );
}

// --- Canvas-rendered histogram ---

function MiniHistogram({ bins, counts, color }: { bins: number[]; counts: number[]; color: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || bins.length === 0) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const maxCount = Math.max(...counts);
    if (maxCount === 0) return;
    const barWidth = w / counts.length;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.6;
    for (let i = 0; i < counts.length; i++) {
      const barH = (counts[i] / maxCount) * (h - 1);
      ctx.fillRect(i * barWidth, h - barH, barWidth - 0.5, barH);
    }
    ctx.globalAlpha = 1;
  }, [bins, counts, color]);

  return (
    <canvas
      ref={canvasRef}
      className="viz-histogram"
    />
  );
}
