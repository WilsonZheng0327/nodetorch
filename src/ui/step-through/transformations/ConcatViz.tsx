// Concat: all inputs as columns with labels + shapes, then output.
// Detects constant tensors (like timestep channels) and shows them as values.

import type { ConcatTransformation } from '../types';
import { FeatureMapsGrid, VectorBars } from './shared';

export function ConcatViz({ t }: { t: ConcatTransformation }) {
  return (
    <div className="tfm-concat">
      <div className="tfm-add-columns">
        {t.inputs.map((inp, i) => (
          <div key={i} className="tfm-add-col">
            <div className="tfm-add-col-label">{inp.label}</div>
            <div className="tfm-concat-shape">{fmtShape(inp.shape)}</div>
            {inp.isConstant ? (
              <div className="tfm-concat-constant">
                <div className="tfm-concat-constant-val">{inp.constantValue?.toFixed(2)}</div>
                <div className="tfm-concat-constant-desc">constant value<br />every pixel</div>
              </div>
            ) : (
              <>
                {inp.featureMaps && <FeatureMapsGrid data={inp.featureMaps} />}
                {inp.vector && <VectorBars values={inp.vector.values} height={100} />}
              </>
            )}
            {i < t.inputs.length - 1 && <div className="tfm-add-op-overlay">|</div>}
          </div>
        ))}
        <div className="tfm-add-col tfm-add-eq">&rarr;</div>
        <div className="tfm-add-col">
          <div className="tfm-add-col-label">Output (dim {t.dim})</div>
          {t.outputShape && <div className="tfm-concat-shape">{fmtShape(t.outputShape)}</div>}
          {t.outputFmaps && <FeatureMapsGrid data={t.outputFmaps} />}
          {t.outputVector && <VectorBars values={t.outputVector.values} height={100} />}
        </div>
      </div>
    </div>
  );
}

function fmtShape(shape: number[]): string {
  return `[${shape.join(', ')}]`;
}
