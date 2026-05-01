// Reshape: before and after with shape labels + visual

import type { ReshapeTransformation } from '../types';
import { FeatureMapsGrid, VectorBars } from './shared';

export function ReshapeViz({ t }: { t: ReshapeTransformation }) {
  const inLabel = `[${t.inputShape.join(', ')}]`;
  const outLabel = `[${t.outputShape.join(', ')}]`;

  return (
    <div className="tfm-reshape">
      {/* Shape calculation */}
      <div className="tfm-flatten-calc">
        <span className="tfm-flatten-calc-before">{inLabel}</span>
        <span className="tfm-flatten-calc-arrow">&rarr;</span>
        <span className="tfm-flatten-calc-after">{outLabel}</span>
      </div>

      {/* Before / After */}
      <div className="tfm-before-after">
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Before</div>
          {t.inputFmaps && <FeatureMapsGrid data={t.inputFmaps} />}
          {t.inputVector && <VectorBars values={t.inputVector.values} height={160} />}
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">After</div>
          {t.outputFmaps && <FeatureMapsGrid data={t.outputFmaps} />}
          {t.outputVector && <VectorBars values={t.outputVector.values} height={160} />}
        </div>
      </div>
    </div>
  );
}
