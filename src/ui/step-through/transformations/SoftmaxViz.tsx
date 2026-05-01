// Softmax transformation: raw logits → probability distribution

import type { SoftmaxTransformation } from '../types';
import { VectorBars, Arrow } from './shared';

export function SoftmaxViz({ t }: { t: SoftmaxTransformation }) {
  const predicted = t.topK[0];

  return (
    <div className="tfm-softmax">
      {predicted && (
        <div className="tfm-softmax-predicted">
          Predicted: <strong>class {predicted.index}</strong> ({(predicted.value * 100).toFixed(1)}%)
        </div>
      )}

      <div className="tfm-flow">
        <div className="tfm-panel">
          <div className="tfm-panel-label">Raw Logits &middot; {t.logits.length} classes</div>
          <VectorBars values={t.logits} height={80} />
        </div>

        <Arrow label="exp + normalize" />

        <div className="tfm-panel">
          <div className="tfm-panel-label">Probabilities &middot; sum = 1</div>
          <VectorBars values={t.probabilities} height={80} />
        </div>
      </div>

      {t.topK.length > 0 && (
        <div className="tfm-section">
          <div className="tfm-section-title">Top Predictions</div>
          <div className="tfm-prob-rows">
            {t.topK.map((p, i) => (
              <div key={i} className="tfm-prob-row">
                <span className="tfm-prob-idx">#{p.index}</span>
                <div className="tfm-prob-bar-bg">
                  <div className="tfm-prob-bar" style={{ width: `${(p.value / (t.topK[0]?.value || 1)) * 100}%` }} />
                </div>
                <span className="tfm-prob-val">{(p.value * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
