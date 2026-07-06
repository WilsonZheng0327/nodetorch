// WiggleWeight — the hero interaction. Sweep one weight of this layer, plot the
// loss as a function of that weight, and show the single most important picture in
// gradient descent: the gradient IS the slope of this curve, and a step walks
// downhill. Drag the slider to wiggle the weight and watch the loss move.

import { useState, useCallback, useEffect } from 'react';
import type { WiggleResult } from './types';
import { apiUrl } from '../../api/base';

const fmt = (v: number | null | undefined, d = 4): string => {
  if (v === null || v === undefined) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e6) return v.toExponential(2);
  return v.toFixed(d);
};
const signed = (v: number | null | undefined, d = 4): string =>
  v === null || v === undefined ? '—' : (v >= 0 ? '+' : '') + fmt(v, d);

/** Linear-interpolate the loss at weight value w from the sampled curve. */
function lossAt(curve: { w: number; loss: number | null }[], w: number): number | null {
  if (curve.length === 0) return null;
  if (w <= curve[0].w) return curve[0].loss;
  if (w >= curve[curve.length - 1].w) return curve[curve.length - 1].loss;
  for (let i = 1; i < curve.length; i++) {
    if (w <= curve[i].w) {
      const a = curve[i - 1];
      const b = curve[i];
      if (a.loss === null || b.loss === null) return a.loss ?? b.loss;
      const t = (w - a.w) / (b.w - a.w || 1);
      return a.loss + t * (b.loss - a.loss);
    }
  }
  return curve[curve.length - 1].loss;
}

/** The prediction at the nearest swept weight value (pred is discrete, so nearest). */
function probeAt(
  curve: { w: number; pTrue?: number | null; pred?: number | null }[],
  w: number,
): { pTrue: number | null; pred: number | null } {
  if (curve.length === 0) return { pTrue: null, pred: null };
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < curve.length; i++) {
    const d = Math.abs(curve[i].w - w);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return { pTrue: curve[best].pTrue ?? null, pred: curve[best].pred ?? null };
}

export function WiggleWeight({
  graphJson,
  nodeId,
  sampleIdx,
}: {
  graphJson: string;
  nodeId: string;
  sampleIdx: number | null;
}) {
  const [data, setData] = useState<WiggleResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragW, setDragW] = useState<number | null>(null);

  const fetchWiggle = useCallback(
    (weightIndex?: number) => {
      setLoading(true);
      setError(null);
      fetch(apiUrl('/wiggle-weight'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          graph: JSON.parse(graphJson),
          nodeId,
          sampleIdx,
          weightIndex,
        }),
      })
        .then((r) => r.json())
        .then((d: { status: string; result?: WiggleResult; error?: string }) => {
          if (d.status === 'ok' && d.result) {
            setData(d.result);
            setDragW(d.result.current.w);
          } else {
            setError(d.error ?? 'Failed to wiggle');
          }
        })
        .catch(() => setError('Cannot connect to backend'))
        .finally(() => setLoading(false));
    },
    [graphJson, nodeId, sampleIdx],
  );

  // Auto-sweep when this layer's cell appears (the component is keyed per layer/
  // sample, so it remounts and re-fetches on navigation). Empty deps → runs once.
  useEffect(() => {
    fetchWiggle();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!data) {
    return (
      <div className="bp-wiggle bp-wiggle-prompt">
        <div className="bp-mech-q">Wiggle a weight — watch the loss move</div>
        {error ? (
          <>
            <div className="backprop-step-error">{error}</div>
            <button className="backprop-step-btn" onClick={() => fetchWiggle()} disabled={loading}>
              Retry
            </button>
          </>
        ) : (
          <div className="bp-note">Sweeping this weight…</div>
        )}
      </div>
    );
  }

  return (
    <WigglePlot
      data={data}
      dragW={dragW ?? data.current.w}
      onDrag={setDragW}
      onAnother={() => fetchWiggle(Math.floor(Math.random() * data.outDim * data.inDim))}
      loading={loading}
    />
  );
}

function WigglePlot({
  data,
  dragW,
  onDrag,
  onAnother,
  loading,
}: {
  data: WiggleResult;
  dragW: number;
  onDrag: (w: number) => void;
  onAnother: () => void;
  loading: boolean;
}) {
  const { curve, current, step, lr, coord, trueLabel, classNames } = data;
  const losses = curve.map((c) => c.loss ?? 0);
  const wMin = curve[0].w;
  const wMax = curve[curve.length - 1].w;
  const lMax = Math.max(...losses) * 1.06 || 1;
  const clsLabel = (c: number | null): string =>
    c == null ? '?' : classNames && c < classNames.length ? classNames[c] : String(c);

  // Plot geometry (viewBox units)
  const W = 620;
  const H = 250;
  const pad = { l: 48, r: 18, t: 18, b: 40 };
  const xOf = (w: number) => pad.l + ((w - wMin) / (wMax - wMin || 1)) * (W - pad.l - pad.r);
  const yOf = (loss: number) => pad.t + (1 - loss / lMax) * (H - pad.t - pad.b);

  const path = curve
    .map((c, i) => `${i === 0 ? 'M' : 'L'} ${xOf(c.w).toFixed(1)} ${yOf(c.loss ?? 0).toFixed(1)}`)
    .join(' ');

  // The tangent line — loss0 + grad·(w − w0). Its slope IS the gradient. We draw
  // it across the whole plot and CLIP to the box, rather than clamping its y
  // (which would bend the line and mis-state the slope).
  const g = current.grad;
  const tangentAt = (w: number) => current.loss + g * (w - current.w);

  // Downhill direction: to lower the loss, move w opposite the gradient's sign.
  const dir = g >= 0 ? -1 : 1;
  const dArr = (wMax - wMin) * 0.18;
  const arrTo = { w: current.w + dir * dArr, l: current.loss - Math.abs(g) * dArr };

  const dragLoss = lossAt(curve, dragW) ?? 0;
  const dragProbe = probeAt(curve, dragW);

  return (
    <div className="bp-wiggle">
      <div className="bp-mech-q">Wiggle a weight — the gradient is the slope of this curve</div>
      <div className="bp-wiggle-sub">
        We take <strong>one</strong> weight of this layer (out {coord[0]}, in {coord[1]}), freeze
        every other weight, and re-measure the loss as we slide it from side to side — that's the
        curve. You're sitting at the <span className="bp-wiggle-key-cur">dot</span>; the curve's{' '}
        <span className="bp-wiggle-key-tan">slope right there is the gradient</span>, and lowering
        the loss means stepping <span className="bp-wiggle-key-down">downhill</span>.
      </div>

      <svg className="bp-wiggle-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        <defs>
          <clipPath id="bp-plot-clip">
            <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} />
          </clipPath>
          <marker id="bp-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#a6e3a1" />
          </marker>
        </defs>

        {/* axes */}
        <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="#45475a" />
        <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="#45475a" />
        <text
          x={(pad.l + W - pad.r) / 2}
          y={H - 8}
          fill="#6c7086"
          fontSize="12"
          textAnchor="middle"
        >
          weight value →
        </text>
        <text
          x={14}
          y={(pad.t + H - pad.b) / 2}
          fill="#6c7086"
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 14 ${(pad.t + H - pad.b) / 2})`}
        >
          loss →
        </text>
        <text x={pad.l - 6} y={pad.t + 4} fill="#6c7086" fontSize="11" textAnchor="end">
          {fmt(lMax, 2)}
        </text>
        <text x={pad.l - 6} y={H - pad.b} fill="#6c7086" fontSize="11" textAnchor="end">
          0
        </text>

        {/* everything data-space is clipped to the plot box so the tangent keeps
            its true slope instead of being clamped/bent at the edges */}
        <g clipPath="url(#bp-plot-clip)">
          <path d={path} fill="none" stroke="#89b4fa" strokeWidth="2" />

          {/* tangent line — its slope IS the gradient */}
          <line
            x1={xOf(wMin)}
            y1={yOf(tangentAt(wMin))}
            x2={xOf(wMax)}
            y2={yOf(tangentAt(wMax))}
            stroke="#f9e2af"
            strokeWidth="1.6"
            strokeDasharray="5 4"
          />
          {/* downhill arrow from the current point */}
          <line
            x1={xOf(current.w)}
            y1={yOf(current.loss)}
            x2={xOf(arrTo.w)}
            y2={yOf(arrTo.l)}
            stroke="#a6e3a1"
            strokeWidth="2.4"
            markerEnd="url(#bp-arrow)"
          />

          {/* dragged point + guide */}
          <line
            x1={xOf(dragW)}
            y1={pad.t}
            x2={xOf(dragW)}
            y2={H - pad.b}
            stroke="#585b70"
            strokeDasharray="2 3"
          />
          <circle
            cx={xOf(dragW)}
            cy={yOf(dragLoss)}
            r="5"
            fill="#cdd6f4"
            stroke="#1e1e2e"
            strokeWidth="1.5"
          />

          {/* current point */}
          <circle cx={xOf(current.w)} cy={yOf(current.loss)} r="4" fill="#f9e2af" />
        </g>
      </svg>

      <div className="bp-wiggle-legend">
        <span className="bp-wiggle-key bp-wiggle-key-cur">● current w = {fmt(current.w)}</span>
        <span className="bp-wiggle-key bp-wiggle-key-tan">
          — slope = ∂L/∂w = {signed(current.grad)}
        </span>
        <span className="bp-wiggle-key bp-wiggle-key-down">→ downhill (−gradient)</span>
      </div>

      <div className="bp-wiggle-slider">
        <div className="bp-wiggle-readout">
          <span>
            drag this weight to <strong>{fmt(dragW)}</strong> → loss{' '}
            <strong>{fmt(dragLoss)}</strong>
          </span>
          {dragProbe.pred != null && (
            <span className="bp-wiggle-pred">
              · model says{' '}
              <strong className={dragProbe.pred === trueLabel ? 'bp-correct' : 'bp-wrong'}>
                {clsLabel(dragProbe.pred)}
              </strong>
              {trueLabel != null && dragProbe.pTrue != null && (
                <span className="bp-wiggle-ptrue">
                  P(true {clsLabel(trueLabel)})
                  <span className="bp-wiggle-ptrue-bar">
                    <span style={{ width: `${dragProbe.pTrue * 100}%` }} />
                  </span>
                  <strong>{(dragProbe.pTrue * 100).toFixed(0)}%</strong>
                </span>
              )}
            </span>
          )}
        </div>
        <input
          type="range"
          min={wMin}
          max={wMax}
          step={(wMax - wMin) / 200}
          value={dragW}
          onChange={(e) => onDrag(parseFloat(e.target.value))}
          className="step-through-slider"
        />
      </div>

      <div className="bp-note">
        The <strong>gradient is the slope</strong> of this curve at the current weight. To shrink
        the loss you move <strong>opposite</strong> the slope — downhill. That's one gradient step:{' '}
        <strong>w ← w − lr·∂L/∂w</strong> = {fmt(current.w)} − {fmt(lr)}·({signed(current.grad)}) ={' '}
        {fmt(step.w)}, and the loss dips {fmt(current.loss)} → {fmt(step.loss)}. Every weight in the
        network moves the same way, at once.
      </div>

      <button className="backprop-sample-btn" onClick={onAnother} disabled={loading}>
        {loading ? 'Sweeping…' : 'Wiggle another weight'}
      </button>
    </div>
  );
}
