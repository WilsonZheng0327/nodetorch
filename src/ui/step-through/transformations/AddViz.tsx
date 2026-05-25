// Add (residual) transformation: inputs + output as columns left to right.
// A[0]  +  B[0]  =  Out[0]
// A[1]     B[1]     Out[1]
// ...

import type { AddTransformation } from '../types';
import { FeatureMapsGrid, VectorBars } from './shared';

export function AddViz({ t }: { t: AddTransformation }) {
  // When every panel is a vector chart (no feature maps), let the columns
  // fill the container width — bar charts read much better at full width.
  const allVectors = t.inputs.every((i) => !i.featureMaps && !!i.vector) && !t.output && !!t.outputVector;
  const barHeight = allVectors ? 180 : 120;

  return (
    <div className="tfm-add">
      <div className={`tfm-add-columns ${allVectors ? 'tfm-add-columns-flex' : ''}`}>
        {t.inputs.map((inp, i) => (
          <div key={i} className="tfm-add-col">
            <div className="tfm-add-col-label">Input {inp.label}</div>
            {inp.featureMaps && <FeatureMapsGrid data={inp.featureMaps} />}
            {inp.vector && <VectorBars values={inp.vector.values} height={barHeight} />}
            {i < t.inputs.length - 1 && <div className="tfm-add-op-overlay">+</div>}
          </div>
        ))}
        <div className="tfm-add-col tfm-add-eq">=</div>
        <div className="tfm-add-col">
          <div className="tfm-add-col-label">Output</div>
          {t.output && <FeatureMapsGrid data={t.output} />}
          {t.outputVector && <VectorBars values={t.outputVector.values} height={barHeight} />}
        </div>
      </div>
    </div>
  );
}
