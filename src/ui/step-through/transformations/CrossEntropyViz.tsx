// CrossEntropy loss: shows predictions, highlights correct class, loss calculation.

import type { CrossEntropyTransformation } from '../types';
import { VectorBars } from './shared';

export function CrossEntropyViz({ t }: { t: CrossEntropyTransformation }) {
  const predicted = t.topK[0];
  const isCorrect = predicted && predicted.index === t.trueLabel;

  return (
    <div className="tfm-cross-entropy">
      {/* Result summary */}
      <div className={`tfm-ce-result ${isCorrect ? 'tfm-ce-correct' : 'tfm-ce-wrong'}`}>
        <div className="tfm-ce-result-label">
          {isCorrect ? 'Correct' : 'Incorrect'}
        </div>
        <div className="tfm-ce-result-detail">
          True class: <strong>{t.classNames ? t.classNames[t.trueLabel] : t.trueLabel}</strong>
          {predicted && !isCorrect && (
            <> &middot; Predicted: <strong>{t.classNames ? t.classNames[predicted.index] : predicted.index}</strong></>
          )}
        </div>
      </div>

      {/* Before/after: logits → probabilities */}
      <div className="tfm-before-after">
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Raw Logits</div>
          <VectorBars values={t.logits} height={160} />
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">After Softmax</div>
          <VectorBars values={t.probabilities} height={160} />
        </div>
      </div>

      {/* Loss calculation breakdown */}
      <div className="tfm-section">
        <div className="tfm-section-title">Loss Calculation</div>
        <div className="tfm-ce-calc">
          <div className="tfm-ce-step">
            <span className="tfm-ce-step-label">P(class {t.trueLabel})</span>
            <span className="tfm-ce-step-value">{t.trueLabelProb.toFixed(4)}</span>
          </div>
          <div className="tfm-ce-step">
            <span className="tfm-ce-step-label">-log(P)</span>
            <span className="tfm-ce-step-value tfm-ce-loss">{t.loss.toFixed(4)}</span>
          </div>
        </div>
      </div>

      {/* Top predictions */}
      <div className="tfm-section">
        <div className="tfm-section-title">Top Predictions</div>
        <div className="tfm-prob-rows">
          {t.topK.map((p, i) => {
            const isTrue = p.index === t.trueLabel;
            return (
              <div key={i} className={`tfm-prob-row ${isTrue ? 'tfm-prob-row-true' : ''}`}>
                <span className="tfm-prob-idx">
                  {t.classNames ? t.classNames[p.index] : `#${p.index}`}
                  {isTrue && <span className="tfm-prob-true-marker"> &#x2713;</span>}
                </span>
                <div className="tfm-prob-bar-bg">
                  <div className="tfm-prob-bar" style={{
                    width: `${(p.value / (t.topK[0]?.value || 1)) * 100}%`,
                    background: isTrue ? '#10b981' : '#89b4fa',
                  }} />
                </div>
                <span className="tfm-prob-val">{(p.value * 100).toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
