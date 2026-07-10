// MechanismViz — plain-language answer to "where does this layer's gradient go?"
// The error at this layer's output is sent backward through its own weights,
// turning into the previous layer's gradient (its "error in"). We show the two
// error vectors and the fact that the weights connect them; the exact arithmetic
// is left out on purpose — the point is the flow, not the numbers.

import type { Mechanism } from './types';
import { signed } from './format';

export function MechanismViz({
  mechanism,
  toName,
}: {
  mechanism: Mechanism;
  toName: string | null;
}) {
  const { outGrad, inGrad, outDim, inDim } = mechanism;
  const prev = toName ?? 'the previous layer';

  return (
    <div className="bp-mech">
      <div className="bp-mech-q">Where the error goes next</div>
      <div className="bp-note bp-mech-explain">
        The gradient here is how wrong this layer's output was. Backprop sends it{' '}
        <strong>backward through this layer's own weights</strong>, which turns it into{' '}
        <strong>{prev}</strong>'s gradient — how wrong <em>its</em> output was. {prev} then repeats
        the exact same step, handing the error one layer further back.
      </div>

      <VecStrip
        label={`error at this layer's output — ∂L/∂output (${outDim})`}
        values={outGrad}
        total={outDim}
      />
      <div className="bp-mech-op">▼ back through the weights (× Wᵀ)</div>
      <VecStrip
        label={`the error ${prev} now sees — ∂L/∂input (${inDim})`}
        values={inGrad}
        total={inDim}
      />
    </div>
  );
}

/** A row of signed values as thin bars diverging up/down from a midline. */
function VecStrip({ label, values, total }: { label: string; values: number[]; total: number }) {
  const max = Math.max(1e-9, ...values.map((v) => Math.abs(v)));
  return (
    <div className="bp-mech-strip">
      <div className="bp-mech-strip-label">{label}</div>
      <div className="bp-mech-bars">
        {values.map((v, i) => {
          const h = (Math.abs(v) / max) * 50;
          const up = v >= 0;
          return (
            <div key={i} className="bp-mech-bar-slot" title={`#${i}: ${signed(v)}`}>
              <div
                className={`bp-mech-bar ${up ? 'up' : 'down'}`}
                style={{ height: `${h}%`, [up ? 'bottom' : 'top']: '50%' } as React.CSSProperties}
              />
            </div>
          );
        })}
        {values.length < total && <span className="bp-mech-ellipsis">…</span>}
      </div>
    </div>
  );
}
