// OneStepBanner — the "before / after one gradient step" summary. Shows what one
// update W ← W − lr·∂L/∂W does to the loss and the prediction on the shown sample.

import type { OneStepResult, StepPrediction } from './types';

const fmt = (v: number | null | undefined, digits = 4): string => {
  if (v === null || v === undefined) return '—';
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential(2);
  return v.toFixed(digits);
};

const classLabel = (idx: number, names: string[] | null): string =>
  names && idx < names.length ? names[idx] : String(idx);

export function OneStepBanner({ result }: { result: OneStepResult }) {
  const { loss, prediction, trueLabel, classNames, lr } = result;

  const lb = loss.before;
  const la = loss.after;
  const lossDropped = lb !== null && la !== null && la < lb;
  const lossPct = lb !== null && la !== null && lb !== 0 ? ((la - lb) / Math.abs(lb)) * 100 : null;

  const before = prediction.before;
  const after = prediction.after;

  return (
    <div className="bp-step-banner">
      <div className="bp-step-title">
        <span className="bp-step-title-main">After one gradient step</span>
        <span className="bp-step-lr">lr {fmt(lr, 4)}</span>
      </div>

      <div className="bp-step-metrics">
        <div className="bp-step-metric">
          <span className="bp-step-metric-label">loss</span>
          <span className="bp-step-metric-val">
            {fmt(lb)} <span className="bp-arrow-sm">→</span> {fmt(la)}
            {lossPct !== null && (
              <span className={`bp-delta ${lossDropped ? 'bp-delta-good' : 'bp-delta-bad'}`}>
                {lossDropped ? '↓' : '↑'} {Math.abs(lossPct).toFixed(1)}%
              </span>
            )}
          </span>
        </div>

        {before && after && (
          <PredictionMetric
            before={before}
            after={after}
            trueLabel={trueLabel}
            classNames={classNames}
          />
        )}
      </div>
    </div>
  );
}

function PredictionMetric({
  before,
  after,
  trueLabel,
  classNames,
}: {
  before: StepPrediction;
  after: StepPrediction;
  trueLabel: number | null;
  classNames: string[] | null;
}) {
  const wasRight = trueLabel !== null && before.predictedClass === trueLabel;
  const nowRight = trueLabel !== null && after.predictedClass === trueLabel;

  const pBefore = before.trueLabelProb;
  const pAfter = after.trueLabelProb;
  const pRose = pBefore !== null && pAfter !== null && pAfter > pBefore;

  return (
    <>
      <div className="bp-step-metric">
        <span className="bp-step-metric-label">
          prediction{trueLabel !== null && <> · true {classLabel(trueLabel, classNames)}</>}
        </span>
        <span className="bp-step-metric-val">
          <span className={wasRight ? 'bp-correct' : 'bp-wrong'}>
            {classLabel(before.predictedClass, classNames)}
          </span>
          <span className="bp-arrow-sm">→</span>
          <span className={nowRight ? 'bp-correct' : 'bp-wrong'}>
            {classLabel(after.predictedClass, classNames)}
          </span>
          {!nowRight && trueLabel !== null && <span className="bp-hint-sm">not there yet</span>}
          {nowRight && !wasRight && (
            <span className="bp-delta bp-delta-good">flipped correct ✓</span>
          )}
        </span>
      </div>

      {trueLabel !== null && (
        <div className="bp-step-metric">
          <span className="bp-step-metric-label">P(true {classLabel(trueLabel, classNames)})</span>
          <span className="bp-step-metric-val">
            {fmt(pBefore)} <span className="bp-arrow-sm">→</span> {fmt(pAfter)}
            {pBefore !== null && pAfter !== null && (
              <span className={`bp-delta ${pRose ? 'bp-delta-good' : 'bp-delta-bad'}`}>
                {pRose ? '↑' : '↓'}
              </span>
            )}
          </span>
        </div>
      )}
    </>
  );
}
