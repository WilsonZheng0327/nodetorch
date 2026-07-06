// LossSeed — the loss node's cell. Backprop starts here, so instead of the
// trivial "grad in = ∂L/∂L = 1" we show the actual starting point: the prediction
// vs. the truth, how the loss is computed, and the error signal ∂L/∂logits (= p − y)
// that every layer's gradient traces back to.

import type { LossSeed as LossSeedData } from './types';

const fmt = (v: number | null | undefined, d = 4): string =>
  v === null || v === undefined ? '—' : v.toFixed(d);

const clsLabel = (i: number, names: string[] | null): string =>
  names && i < names.length ? names[i] : `#${i}`;

export function LossSeed({ seed, toName }: { seed: LossSeedData; toName?: string | null }) {
  const { probabilities, errorSignal, trueLabel, trueLabelProb, loss, classNames, topK } = seed;
  const prev = toName ?? 'the last layer';

  // Which classes to show: the top predictions plus the true class.
  const idxSet = new Set<number>(topK.map((t) => t.index));
  if (trueLabel != null) idxSet.add(trueLabel);
  const rows = [...idxSet].sort((a, b) => (probabilities[b] ?? 0) - (probabilities[a] ?? 0));
  const maxErr = Math.max(1e-9, ...errorSignal.map((e) => Math.abs(e)));

  const predicted = probabilities.indexOf(Math.max(...probabilities));
  const correct = trueLabel != null && predicted === trueLabel;

  return (
    <div className="stage-detail">
      <div className="stage-detail-header">
        <span className="stage-detail-name">Loss — where backprop starts</span>
        <span className="stage-detail-shape">
          loss <strong className="bp-loss-big">{fmt(loss)}</strong>
        </span>
      </div>

      <div className="bp-seed-note">
        <div className="bp-seed-question">
          The whole backward pass answers one question:{' '}
          <strong>how should each layer change to make this loss smaller?</strong> It starts here.
        </div>
        The loss compares the prediction to the truth. Its gradient w.r.t. the logits —{' '}
        <strong>∂L/∂logits = predicted − target</strong> — is the error signal every layer's update
        traces back to. Below: first <em>what</em> the error is, then <em>how it reaches {prev}</em>
        .
      </div>

      <div className="bp-seed-cols">
        {/* Prediction vs truth */}
        <div className="bp-panel">
          <div className="bp-panel-title">
            Prediction{' '}
            {trueLabel != null && (
              <span className={correct ? 'bp-correct' : 'bp-wrong'}>
                · {correct ? 'correct' : 'wrong'}
              </span>
            )}
          </div>
          <div className="bp-seed-rows">
            {rows.map((i) => {
              const isTrue = i === trueLabel;
              return (
                <div key={i} className={`bp-seed-row ${isTrue ? 'bp-seed-row-true' : ''}`}>
                  <span className="bp-seed-label">
                    {clsLabel(i, classNames)}
                    {isTrue && ' ✓'}
                  </span>
                  <div className="bp-seed-bar-bg">
                    <div
                      className="bp-seed-bar"
                      style={{
                        width: `${(probabilities[i] ?? 0) * 100}%`,
                        background: isTrue ? '#a6e3a1' : '#89b4fa',
                      }}
                    />
                  </div>
                  <span className="bp-seed-pct">{((probabilities[i] ?? 0) * 100).toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
          <div className="bp-note">
            P(true {trueLabel != null ? clsLabel(trueLabel, classNames) : '?'}) ={' '}
            <strong>{fmt(trueLabelProb)}</strong> → loss = −log(P) = <strong>{fmt(loss)}</strong>
          </div>
        </div>

        {/* Error signal ∂L/∂logits = p − y */}
        <div className="bp-panel">
          <div className="bp-panel-title">Error signal — ∂L/∂logits = p − y</div>
          <div className="bp-seed-rows">
            {rows.map((i) => {
              const e = errorSignal[i] ?? 0;
              const isTrue = i === trueLabel;
              const push = e < 0 ? 'up' : 'down';
              return (
                <div key={i} className="bp-seed-row">
                  <span className="bp-seed-label">{clsLabel(i, classNames)}</span>
                  <div className="bp-err-track">
                    <div className="bp-err-mid" />
                    <div
                      className={`bp-err-bar bp-err-${push}`}
                      style={{ width: `${(Math.abs(e) / maxErr) * 50}%` }}
                    />
                  </div>
                  <span className={`bp-seed-pct ${isTrue ? 'bp-correct' : ''}`}>{fmt(e, 3)}</span>
                </div>
              );
            })}
          </div>
          <div className="bp-note">
            Negative pushes a logit <strong>up</strong>, positive pushes it <strong>down</strong>.
            The true class is the only one we want higher — everything else, lower.
          </div>
        </div>
      </div>

      <div className="bp-seed-bridge">
        <div className="bp-seed-bridge-q">→ So how does this error change {prev}?</div>
        <div className="bp-note">
          This <strong>p − y</strong> vector is exactly the gradient that arrives at{' '}
          <strong>{prev}</strong> — its ∂L/∂output. Step left (←) to see it routed back through that
          layer's weights, becoming a weight update and the error for the layer before it.
        </div>
      </div>
    </div>
  );
}
