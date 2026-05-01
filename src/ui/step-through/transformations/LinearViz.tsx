// Linear transformation:
// - Matrix notation: [1, out] = [1, in] @ [in, out] + [1, out]
// - Before/after value bars

import type { LinearTransformation } from '../types';
import { VectorBars } from './shared';

export function LinearViz({ t }: { t: LinearTransformation }) {
  return (
    <div className="tfm-linear">
      {/* Matrix equation with labels */}
      <div className="tfm-linear-diagram">
        <div className="tfm-linear-eq-line">
          <div className="tfm-linear-term">
            <span className="tfm-linear-label">output</span>
            <span className="tfm-linear-shape tfm-linear-output-color">[1, {t.outputDim}]</span>
          </div>
          <span className="tfm-linear-op">=</span>
          <div className="tfm-linear-term">
            <span className="tfm-linear-label">input</span>
            <span className="tfm-linear-shape tfm-linear-input-color">[1, {t.inputDim}]</span>
          </div>
          <span className="tfm-linear-op">@</span>
          <div className="tfm-linear-term">
            <span className="tfm-linear-label">weights</span>
            <span className="tfm-linear-shape tfm-linear-w-color">[{t.inputDim}, {t.outputDim}]</span>
          </div>
          <span className="tfm-linear-op">+</span>
          <div className="tfm-linear-term">
            <span className="tfm-linear-label">bias</span>
            <span className="tfm-linear-shape tfm-linear-w-color">[1, {t.outputDim}]</span>
          </div>
        </div>
      </div>

      {/* Before/after values */}
      <div className="tfm-before-after">
        <div className="tfm-ba-pane">
          <VectorBars values={t.inputVector} height={200} label={`Before \u00b7 ${t.inputDim} values`} />
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <VectorBars values={t.outputVector} height={200} label={`After \u00b7 ${t.outputDim} values`} />
        </div>
      </div>
    </div>
  );
}
