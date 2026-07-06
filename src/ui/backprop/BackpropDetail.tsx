// BackpropDetail — the per-layer "backward cell", read right → left to match the
// backward flow: GRAD IN (∂L/∂output, from the next layer) → THE UPDATE (∂L/∂W,
// the biggest panel — this is what learning IS) → GRAD OUT (∂L/∂input, passed to
// the previous layer, which is literally its grad-in). The loss node is special-
// cased into LossSeed (see below). Distribution stats are tucked behind an
// expander so the change stays front and center.

import type { BackwardStage, BackwardSample, GradStats, ParamGradStats, StepLayer } from './types';
import { LossSeed } from './LossSeed';
import { MechanismViz } from './MechanismViz';
import { WiggleWeight } from './WiggleWeight';
import { SamplePreview } from './SamplePreview';

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
  showAfter = false,
  fromName = null,
  toName = null,
  graphJson,
  sampleIdx = null,
  sample = null,
}: {
  stage: BackwardStage;
  stepLayer?: StepLayer | null;
  lr?: number | null;
  showAfter?: boolean;
  fromName?: string | null;
  toName?: string | null;
  graphJson?: string;
  sampleIdx?: number | null;
  sample?: BackwardSample | null;
}) {
  // The synthetic data node at the left of the timeline — show the input sample.
  if (stage.nodeType?.startsWith('data')) return <DataDetail sample={sample} />;

  // The loss node is where backprop starts — show the seed, not a generic cell.
  if (stage.lossSeed) return <LossSeed seed={stage.lossSeed} toName={toName} />;

  const shapeIn = stage.gradientShape ?? stage.outputShape;

  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">
          {stage.blockName && <span className="stage-detail-block">{stage.blockName}</span>}
          {stage.displayName}
        </span>
        <span className="stage-detail-shape">{shapeIn ? `[${shapeIn.join(', ')}]` : '—'}</span>
      </div>

      {stage.insight && <div className="stage-detail-insight">{stage.insight}</div>}

      <div className="stage-detail-body">
        <div className="bp-cell">
          <GradFlow
            side="out"
            stats={stage.gradOut?.stats}
            shape={stage.gradOut?.shape}
            neighbor={toName}
          />
          <div className="bp-arrow">←</div>
          <UpdatePanel
            pgs={stage.paramGradStats}
            stepLayer={stepLayer}
            lr={lr}
            showAfter={showAfter}
          />
          <div className="bp-arrow">←</div>
          <GradFlow side="in" stats={stage.stats} shape={shapeIn} neighbor={fromName} />
        </div>

        {stage.mechanism && (
          <div className="bp-lower">
            <MechanismViz mechanism={stage.mechanism} toName={toName} />
            {graphJson && (
              <WiggleWeight
                key={`${stage.nodeId}-${sampleIdx}`}
                graphJson={graphJson}
                nodeId={stage.nodeId}
                sampleIdx={sampleIdx}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/** The synthetic data node's detail — the input sample the gradients were for. */
function DataDetail({ sample }: { sample: BackwardSample | null }) {
  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">Data — the input</span>
        <span className="stage-detail-shape">where backprop ends</span>
      </div>
      {sample && <SamplePreview sample={sample} />}
      <div className="bp-note bp-data-note">
        This is the sample the gradients were computed for. Backprop starts at the loss (far right)
        and traces the error back through every layer to here. Use <strong>change</strong> on this
        card to draw a new sample{sample?.classNames ? ' or pick one by class' : ''}.
      </div>
    </div>
  );
}

/** Slim grad-flow panel (the two sides). Leads with a plain description of where
 * the gradient goes; the magnitude/distribution numbers hide behind an expander.
 * `in` = ∂L/∂output (right, from downstream), `out` = ∂L/∂input (left, to upstream). */
function GradFlow({
  side,
  stats,
  shape,
  neighbor,
}: {
  side: 'in' | 'out';
  stats?: GradStats;
  shape?: number[];
  neighbor: string | null;
}) {
  const isIn = side === 'in';
  return (
    <div className="bp-panel bp-panel-side">
      <div className="bp-panel-title">
        {isIn ? 'Gradient in — ∂L/∂output' : 'Gradient out — ∂L/∂input'}
      </div>
      <div className="bp-note">
        {isIn ? (
          <>
            Arrives from <strong>{neighbor ?? 'the loss'}</strong> (the layer to the right).
          </>
        ) : neighbor ? (
          <>
            Passed back to <strong>{neighbor}</strong> — it becomes that layer’s gradient in.
          </>
        ) : (
          <>Reaches the network input.</>
        )}
      </div>
      {stats && (
        <details className="bp-dist">
          <summary>magnitude</summary>
          <div className="bp-norm-sm">
            ‖grad‖ {fmt(stats.norm)}
            {shape ? ` · [${shape.join(', ')}]` : ''}
          </div>
          <div className="bp-stat-grid">
            <Stat label="mean" value={stats.mean} />
            <Stat label="std" value={stats.std} />
            <Stat label="min" value={stats.min} />
            <Stat label="max" value={stats.max} />
          </div>
        </details>
      )}
    </div>
  );
}

/** The middle, biggest panel — the actual weight update (or a passthrough note). */
function UpdatePanel({
  pgs,
  stepLayer,
  lr,
  showAfter,
}: {
  pgs?: ParamGradStats;
  stepLayer?: StepLayer | null;
  lr?: number | null;
  showAfter: boolean;
}) {
  if (!pgs) {
    return (
      <div className="bp-panel bp-panel-mid">
        <div className="bp-panel-title">The update — ∂L/∂W</div>
        <div className="bp-passthrough">no learnable weights</div>
        <div className="bp-note">
          Nothing to update here — this layer just reshapes the gradient and passes it on. Compare
          ‖grad‖ on the right vs. the left to see how it changed the signal.
        </div>
      </div>
    );
  }

  const showChange = showAfter && stepLayer;

  return (
    <div className="bp-panel bp-panel-mid">
      <div className="bp-panel-title">The update — ∂L/∂W</div>

      {showChange ? (
        // After a step: lead with the realized weight change.
        <>
          <div className="bp-update-hero">
            <div className="bp-update-hero-label">weights moved</div>
            <div className="bp-update-hero-val">
              ‖W‖ {fmt(stepLayer!.weightNormBefore)} <span className="bp-arrow-sm">→</span>{' '}
              {fmt(stepLayer!.weightNormAfter)}
            </div>
            <div className="bp-update-delta">
              largest single weight moved <strong>{fmt(stepLayer!.maxAbsDelta)}</strong>
            </div>
          </div>
          <div className="bp-note">
            Every weight stepped by <strong>−lr · ∂L/∂W</strong>
            {lr != null && <> (lr {fmt(lr)})</>} — downhill on the loss.
          </div>
        </>
      ) : (
        // Before a step: lead with the pending nudge (the gradient's magnitude).
        <>
          <div className="bp-update-hero">
            <div className="bp-update-hero-label">gradient pushing the weights</div>
            <div className="bp-update-hero-val">‖∂L/∂W‖ {fmt(pgs.gradNorm)}</div>
            <div className="bp-update-delta">
              <strong>W ← W − lr · ∂L/∂W</strong>
            </div>
          </div>
          <HealthBadge health={pgs.health} />
          <div className="bp-note">Take a step to see the weights actually move.</div>
        </>
      )}

      <details className="bp-dist">
        <summary>weight ↔ gradient sizes</summary>
        <div className="bp-stat-grid">
          <Stat label="‖∂L/∂W‖" value={pgs.gradNorm} />
          <Stat label="‖W‖" value={pgs.weightNorm} />
          <Stat label="ratio" value={pgs.ratio} />
        </div>
      </details>
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
