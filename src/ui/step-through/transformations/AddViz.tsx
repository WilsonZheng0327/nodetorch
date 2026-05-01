// Add (residual) transformation: inputs + output as columns left to right.
// A[0]  +  B[0]  =  Out[0]
// A[1]     B[1]     Out[1]
// ...

import type { AddTransformation } from '../types';
import { FeatureMapsGrid, VectorBars } from './shared';

export function AddViz({ t }: { t: AddTransformation }) {
  return (
    <div className="tfm-add">
      <div className="tfm-add-columns">
        {t.inputs.map((inp, i) => (
          <div key={i} className="tfm-add-col">
            <div className="tfm-add-col-label">Input {inp.label}</div>
            {inp.featureMaps && <FeatureMapsGrid data={inp.featureMaps} />}
            {inp.vector && <VectorBars values={inp.vector.values} height={120} />}
            {i < t.inputs.length - 1 && <div className="tfm-add-op-overlay">+</div>}
          </div>
        ))}
        <div className="tfm-add-col tfm-add-eq">=</div>
        <div className="tfm-add-col">
          <div className="tfm-add-col-label">Output</div>
          {t.output && <FeatureMapsGrid data={t.output} />}
          {t.outputVector && <VectorBars values={t.outputVector.values} height={120} />}
        </div>
      </div>
    </div>
  );
}
