// BackpropDetail — the per-layer "backward cell". For the selected stage it shows
// the two tensors the backward pass produces here:
//   1. the gradient arriving from downstream (dL/d-output) — norm + distribution
//   2. the weight-gradient it produces (dL/d-weights) — its size relative to the
//      weights themselves, i.e. how big a nudge this step wants to make.
// Once a gradient step is taken, the weight panel also shows ‖W‖ before → after.

import type { BackwardStage, GradStats, ParamGradStats, StepLayer } from './types';

// Gradients are often tiny (1e-5 and smaller). Plain toFixed(4) collapses them
// all to "0.0000", so fall back to scientific notation for small nonzero values.
const fmt = (v: number | null | undefined, digits = 4): string => {
  if (v === null || v === undefined) return '—';
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential(2);
  return v.toFixed(digits);
};

export function BackpropDetail({
  stage,
  stepLayer,
  lr,
}: {
  stage: BackwardStage;
  stepLayer?: StepLayer | null;
  lr?: number | null;
}) {
  const { stats, paramGradStats } = stage;
  const shape = stage.gradientShape ?? stage.outputShape;

  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">
          {stage.blockName && <span className="stage-detail-block">{stage.blockName}</span>}
          {stage.displayName}
        </span>
        <span className="stage-detail-shape">{shape ? `grad [${shape.join(', ')}]` : '—'}</span>
      </div>

      {stage.insight && <div className="stage-detail-insight">{stage.insight}</div>}

      <div className="stage-detail-body">
        <div className="bp-cell">
          <GradientArriving stats={stats} shape={shape} />
          <div className="bp-arrow">→</div>
          <WeightGradient pgs={paramGradStats} stepLayer={stepLayer} lr={lr} />
        </div>
      </div>
    </div>
  );
}

function GradientArriving({ stats, shape }: { stats?: GradStats; shape?: number[] }) {
  return (
    <div className="bp-panel">
      <div className="bp-panel-title">Gradient in — ∂L/∂output</div>
      {stats ? (
        <>
          <div className="bp-norm">
            ‖grad‖ <strong>{fmt(stats.norm)}</strong>
          </div>
          <div className="bp-stat-grid">
            <Stat label="mean" value={stats.mean} />
            <Stat label="std" value={stats.std} />
            <Stat label="min" value={stats.min} />
            <Stat label="max" value={stats.max} />
          </div>
          <div className="bp-note">
            The gradient of the loss w.r.t. this layer's output
            {shape ? `, shape [${shape.join(', ')}]` : ''}. It arrives from the layer to the right
            and drives the update here.
          </div>
        </>
      ) : (
        <div className="bp-note">No gradient reached this layer.</div>
      )}
    </div>
  );
}

function WeightGradient({
  pgs,
  stepLayer,
  lr,
}: {
  pgs?: ParamGradStats;
  stepLayer?: StepLayer | null;
  lr?: number | null;
}) {
  if (!pgs) {
    return (
      <div className="bp-panel">
        <div className="bp-panel-title">Weight gradient — ∂L/∂W</div>
        <div className="bp-note">
          This layer has no learnable weights — it just passes the gradient through.
        </div>
      </div>
    );
  }
  return (
    <div className="bp-panel">
      <div className="bp-panel-title">Weight gradient — ∂L/∂W</div>
      <div className="bp-norm">
        ‖∂L/∂W‖ <strong>{fmt(pgs.gradNorm)}</strong>
      </div>
      <div className="bp-stat-grid">
        <Stat label="‖W‖" value={pgs.weightNorm} />
        <Stat label="ratio" value={pgs.ratio} digits={4} />
      </div>
      <HealthBadge health={pgs.health} />
      {stepLayer ? (
        <div className="bp-step-weight">
          <div className="bp-step-weight-row">
            <span className="bp-step-metric-label">‖W‖ after one step</span>
            <span className="bp-step-metric-val">
              {fmt(stepLayer.weightNormBefore)} <span className="bp-arrow-sm">→</span>{' '}
              {fmt(stepLayer.weightNormAfter)}
            </span>
          </div>
          <div className="bp-note">
            Largest single-weight move this step: <strong>{fmt(stepLayer.maxAbsDelta)}</strong>
            {lr != null && <> (lr {fmt(lr)})</>}.
          </div>
        </div>
      ) : (
        <div className="bp-note">
          The update wants <strong>W ← W − lr · ∂L/∂W</strong>. The ratio (gradient size vs. weight
          size) tells you how big a relative nudge that is.
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  digits = 4,
}: {
  label: string;
  value: number | null;
  digits?: number;
}) {
  return (
    <div className="bp-stat">
      <span className="bp-stat-label">{label}</span>
      <span className="bp-stat-value">{fmt(value, digits)}</span>
    </div>
  );
}

function HealthBadge({ health }: { health: string }) {
  const kind =
    health === 'healthy'
      ? 'ok'
      : health === 'vanishing'
        ? 'low'
        : health === 'exploding'
          ? 'high'
          : 'neutral';
  return <span className={`bp-health bp-health-${kind}`}>{health}</span>;
}
