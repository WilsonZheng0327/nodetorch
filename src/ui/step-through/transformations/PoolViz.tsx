// Pool transformation:
// - Before/after feature maps (half and half)
// - Interactive pool step-through: move pool window across input,
//   see max/avg selection, result highlighted in output

import { useState, useRef, useEffect } from 'react';
import type { PoolTransformation } from '../types';
import { FeatureMapsGrid, VectorBars } from './shared';

const KIND_LABELS: Record<string, string> = {
  max: 'Max Pool', avg: 'Avg Pool', adaptive_avg: 'Adaptive Avg Pool',
};

export function PoolViz({ t }: { t: PoolTransformation }) {
  const hasInteractive = t.rawInput && t.rawOutput && t.poolSize;
  const [pos, setPos] = useState<{ r: number; c: number }>({ r: 0, c: 0 });

  const poolH = t.poolSize?.[0] ?? 2;
  const poolW = t.poolSize?.[1] ?? 2;
  const strideH = t.strideSize?.[0] ?? poolH;
  const strideW = t.strideSize?.[1] ?? poolW;
  const outH = t.output.height;
  const outW = t.output.width;

  const posR = Math.min(pos.r, Math.max(0, outH - 1));
  const posC = Math.min(pos.c, Math.max(0, outW - 1));
  const inR = posR * strideH;
  const inC = posC * strideW;

  return (
    <div className="tfm-pool">
      {/* Before / After split */}
      <div className="tfm-before-after">
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">Before &middot; {t.input.channels} ch &middot; {t.input.height}&times;{t.input.width}</div>
          <FeatureMapsGrid data={t.input} />
        </div>
        <div className="tfm-ba-divider" />
        <div className="tfm-ba-pane">
          <div className="tfm-ba-label">After &middot; {t.output.channels} ch &middot; {t.output.height}&times;{t.output.width}</div>
          {t.channelValues ? (
            <PooledValues values={t.channelValues} channels={t.output.channels} />
          ) : t.output.height <= 2 && t.output.width <= 2 ? (
            <div className="tfm-note">Each channel collapsed to a single value</div>
          ) : (
            <FeatureMapsGrid data={t.output} />
          )}
        </div>
      </div>

      {/* Interactive step-through */}
      {hasInteractive && (
        <div className="tfm-section">
          <div className="tfm-section-title">
            Step Through &middot; {KIND_LABELS[t.poolKind]} {poolH}&times;{poolW}, stride {strideH}
          </div>
          <PoolStepThrough
            rawInput={t.rawInput!} rawOutput={t.rawOutput!}
            inH={t.rawInputH!} inW={t.rawInputW!}
            outH={outH} outW={outW}
            poolH={poolH} poolW={poolW}
            strideH={strideH} strideW={strideW}
            poolKind={t.poolKind}
            posR={posR} posC={posC} inR={inR} inC={inC}
            onMove={(r, c) => setPos({ r, c })}
          />
        </div>
      )}
    </div>
  );
}

function PoolStepThrough({ rawInput, rawOutput, inH, inW, outH, outW, poolH, poolW, strideH, strideW, poolKind, posR, posC, inR, inC, onMove }: {
  rawInput: number[][]; rawOutput: number[][];
  inH: number; inW: number; outH: number; outW: number;
  poolH: number; poolW: number; strideH: number; strideW: number;
  poolKind: string;
  posR: number; posC: number; inR: number; inC: number;
  onMove: (r: number, c: number) => void;
}) {
  // Extract pool window values
  const windowValues: { val: number; r: number; c: number }[] = [];
  let maxVal = -Infinity;
  let maxIdx = 0;
  let sum = 0;
  for (let pr = 0; pr < poolH; pr++) {
    for (let pc = 0; pc < poolW; pc++) {
      const ir = inR + pr;
      const ic = inC + pc;
      const val = (ir >= 0 && ir < inH && ic >= 0 && ic < inW) ? rawInput[ir][ic] : 0;
      const idx = windowValues.length;
      windowValues.push({ val, r: ir, c: ic });
      sum += val;
      if (val > maxVal) { maxVal = val; maxIdx = idx; }
    }
  }
  const outputVal = rawOutput[posR]?.[posC] ?? 0;
  const isMax = poolKind === 'max';

  return (
    <div className="tfm-conv-step">
      {/* Input with pool window */}
      <div className="tfm-conv-step-panel">
        <div className="tfm-conv-step-label">Input (ch 0)</div>
        <GridCanvas
          rawData={rawInput} rows={inH} cols={inW}
          highlightR={inR} highlightC={inC} highlightH={poolH} highlightW={poolW}
          highlightColor="#a6e3a1"
          onClick={(r, c) => {
            const outR = Math.floor(r / strideH);
            const outC = Math.floor(c / strideW);
            onMove(Math.max(0, Math.min(outH - 1, outR)), Math.max(0, Math.min(outW - 1, outC)));
          }}
        />
      </div>

      {/* Computation */}
      <div className="tfm-conv-step-calc">
        <div className="tfm-conv-step-label">{poolH}&times;{poolW} Window</div>
        <div className="tfm-conv-calc-grid">
          {Array.from({ length: poolH }, (_, pr) => (
            <div key={pr} className="tfm-conv-calc-row">
              {Array.from({ length: poolW }, (_, pc) => {
                const idx = pr * poolW + pc;
                const wv = windowValues[idx];
                const isSelected = isMax && idx === maxIdx;
                return (
                  <div key={pc} className={`tfm-pool-calc-cell ${isSelected ? 'tfm-pool-calc-selected' : ''}`}>
                    {fmtV(wv.val)}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className="tfm-conv-calc-result">
          {isMax ? (
            <span>max = <strong>{fmtV(maxVal)}</strong></span>
          ) : (
            <span>avg = ({windowValues.map(w => fmtV(w.val)).join(' + ')}) / {windowValues.length} = <strong>{fmtV(outputVal)}</strong></span>
          )}
        </div>
        <div className="tfm-conv-calc-pos">output[{posR}, {posC}]</div>
      </div>

      {/* Output with highlighted pixel */}
      <div className="tfm-conv-step-panel">
        <div className="tfm-conv-step-label">Output (ch 0)</div>
        <GridCanvas
          rawData={rawOutput} rows={outH} cols={outW}
          highlightR={posR} highlightC={posC} highlightH={1} highlightW={1}
          highlightColor="#89b4fa"
          onClick={(r, c) => onMove(Math.max(0, Math.min(outH - 1, r)), Math.max(0, Math.min(outW - 1, c)))}
        />
      </div>
    </div>
  );
}

/** Grayscale grid canvas with a highlight rectangle. Clickable. */
function GridCanvas({ rawData, rows, cols, highlightR, highlightC, highlightH, highlightW, highlightColor, onClick }: {
  rawData: number[][]; rows: number; cols: number;
  highlightR: number; highlightC: number; highlightH: number; highlightW: number;
  highlightColor: string;
  onClick: (r: number, c: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cellSize = Math.max(4, Math.min(12, Math.floor(220 / Math.max(rows, cols))));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = cols * cellSize;
    const h = rows * cellSize;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const vals = rawData.flat();
    const vmin = Math.min(...vals);
    const vmax = Math.max(...vals);
    const vrange = vmax - vmin || 1;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.round((((rawData[r]?.[c] ?? 0) - vmin) / vrange) * 255);
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
      }
    }

    // Highlight
    ctx.strokeStyle = highlightColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(
      highlightC * cellSize + 1, highlightR * cellSize + 1,
      highlightW * cellSize - 2, highlightH * cellSize - 2,
    );
  }, [rawData, rows, cols, highlightR, highlightC, highlightH, highlightW, highlightColor, cellSize]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / rect.width * cols);
    const r = Math.floor((e.clientY - rect.top) / rect.height * rows);
    onClick(Math.max(0, Math.min(rows - 1, r)), Math.max(0, Math.min(cols - 1, c)));
  };

  return (
    <canvas
      ref={canvasRef}
      className="tfm-conv-step-canvas"
      style={{ width: cols * cellSize, height: rows * cellSize, cursor: 'crosshair' }}
      onClick={handleClick}
    />
  );
}

/** When output is 1x1, show actual per-channel values as a bar chart. */
function PooledValues({ values, channels }: { values: number[]; channels: number }) {
  return (
    <div className="tfm-pool-values">
      <div className="tfm-pool-values-label">{values.length} of {channels} channels &mdash; one value per channel</div>
      <VectorBars values={values} height={160} label="Per-channel pooled values" />
    </div>
  );
}

function fmtV(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}
