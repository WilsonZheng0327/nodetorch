// Conv2d / ConvTranspose2d transformation:
// - Before/after feature maps split
// - Interactive step-through:
//   Conv2d: click output pixel → see which input patch + kernel produced it
//   ConvTranspose2d: click input pixel → see how kernel scatters it to output region

import { useState, useRef, useEffect } from 'react';
import type { Conv2dTransformation } from '../types';
import { FeatureMapsGrid, GrayscaleCanvas } from './shared';

export function Conv2dViz({ t }: { t: Conv2dTransformation }) {
  const hasInteractive = t.rawInputs && t.allKernels && t.rawOutputs;
  const [selectedFilter, setSelectedFilter] = useState(0);
  const [selectedInCh, setSelectedInCh] = useState(0);
  const [pos, setPos] = useState<{ r: number; c: number }>({ r: 0, c: 0 });

  const kH = t.kernels?.kernelH ?? 3;
  const kW = t.kernels?.kernelW ?? 3;
  const stride = t.stride ?? [1, 1];
  const padding = t.padding ?? [0, 0];
  const inH = t.rawInputH ?? t.input.height;
  const inW = t.rawInputW ?? t.input.width;
  const outH = t.output.height;
  const outW = t.output.width;
  const numInCh = t.rawInputs?.length ?? 1;
  const isTranspose = t.isTranspose ?? false;

  return (
    <div className="tfm-conv2d">
      {/* Filter selector */}
      {t.kernels && (
        <div className="tfm-section">
          <div className="tfm-section-title">
            Filters &middot; {t.kernels.showing} of {t.kernels.totalFilters} &middot; {kH}&times;{kW}
          </div>
          <div className="tfm-kernel-grid">
            {t.kernels.data.map((k, i) => (
              <div key={i} className={`tfm-kernel-selectable ${i === selectedFilter ? 'tfm-kernel-selected' : ''}`} onClick={() => setSelectedFilter(i)}>
                <GrayscaleCanvas pixels={k} size={40} />
                <span className="tfm-fmap-label">{i}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Interactive step-through */}
      {hasInteractive && (
        isTranspose ? (
          <TransposeConvInteractive
            rawInputs={t.rawInputs!} allKernels={t.allKernels!}
            rawOutput={t.rawOutputs![selectedFilter] ?? t.rawOutputs![0]}
            inH={inH} inW={inW} outH={outH} outW={outW}
            kH={kH} kW={kW} stride={stride} padding={padding}
            selectedFilter={selectedFilter} selectedInCh={selectedInCh}
            numInCh={numInCh}
            posR={pos.r} posC={pos.c}
            onMove={(r, c) => setPos({ r, c })}
            onSelectInCh={setSelectedInCh}
          />
        ) : (
          <ConvInteractive
            rawInputs={t.rawInputs!} allKernels={t.allKernels!}
            rawOutput={t.rawOutputs![selectedFilter] ?? t.rawOutputs![0]}
            bias={t.biases?.[selectedFilter] ?? 0}
            inH={inH} inW={inW} outH={outH} outW={outW}
            kH={kH} kW={kW} stride={stride} padding={padding}
            selectedFilter={selectedFilter} selectedInCh={selectedInCh}
            numInCh={numInCh}
            posR={Math.min(pos.r, Math.max(0, outH - 1))}
            posC={Math.min(pos.c, Math.max(0, outW - 1))}
            onMove={(r, c) => setPos({ r, c })}
            onSelectInCh={setSelectedInCh}
          />
        )
      )}
    </div>
  );
}

// ============================================================================
// Regular Conv2d interactive
// ============================================================================

function ConvInteractive({ rawInputs, allKernels, rawOutput, bias, inH, inW, outH, outW, kH, kW, stride, padding,
  selectedFilter, selectedInCh, numInCh, posR, posC, onMove, onSelectInCh }: {
  rawInputs: number[][][][]; allKernels: number[][][][]; rawOutput: number[][];
  bias: number; inH: number; inW: number; outH: number; outW: number;
  kH: number; kW: number; stride: number[]; padding: number[];
  selectedFilter: number; selectedInCh: number; numInCh: number;
  posR: number; posC: number;
  onMove: (r: number, c: number) => void; onSelectInCh: (ch: number) => void;
}) {
  const inR = posR * stride[0] - padding[0];
  const inC = posC * stride[1] - padding[1];
  const rawInput = rawInputs[selectedInCh] ?? rawInputs[0];
  const kernel = allKernels[selectedFilter]?.[selectedInCh] ?? allKernels[0]?.[0];

  // Per-channel partial sums
  const perChannelSums: number[] = [];
  for (let ch = 0; ch < numInCh; ch++) {
    let chSum = 0;
    const chInput = rawInputs[ch] ?? rawInputs[0];
    const chKernel = allKernels[selectedFilter]?.[ch] ?? allKernels[0]?.[0];
    for (let kr = 0; kr < kH; kr++) for (let kc = 0; kc < kW; kc++) {
      const ir = inR + kr, ic = inC + kc;
      const inVal = (ir >= 0 && ir < inH && ic >= 0 && ic < inW) ? (chInput[ir]?.[ic] ?? 0) : 0;
      chSum += inVal * (chKernel?.[kr]?.[kc] ?? 0);
    }
    perChannelSums.push(chSum);
  }
  const totalSum = perChannelSums.reduce((a, b) => a + b, 0);
  const outputVal = rawOutput[posR]?.[posC] ?? 0;

  // Current channel patch
  const patchValues: { inVal: number; kVal: number }[] = [];
  for (let kr = 0; kr < kH; kr++) for (let kc = 0; kc < kW; kc++) {
    const ir = inR + kr, ic = inC + kc;
    const inVal = (ir >= 0 && ir < inH && ic >= 0 && ic < inW) ? (rawInput[ir]?.[ic] ?? 0) : 0;
    patchValues.push({ inVal, kVal: kernel?.[kr]?.[kc] ?? 0 });
  }
  const currentChSum = perChannelSums[selectedInCh] ?? 0;

  return (
    <div className="tfm-section">
      <div className="tfm-section-title">Step Through &middot; Filter {selectedFilter} &middot; output[{posR}, {posC}]</div>
      <div className="tfm-conv-formula">
        output[{posR},{posC}] = {numInCh > 1 && <>&Sigma;<sub>ch</sub> </>}(input * kernel){bias !== 0 ? ' + bias' : ''} = <strong>{fmtV(outputVal)}</strong>
      </div>
      {numInCh > 1 && <ChannelSelector numInCh={numInCh} selectedInCh={selectedInCh} onSelectInCh={onSelectInCh} />}
      <div className="tfm-conv-step">
        <div className="tfm-conv-step-panel">
          <div className="tfm-conv-step-label">Input ch {selectedInCh}</div>
          <GridWithOverlay rawData={rawInput} rows={inH} cols={inW} overlayR={inR} overlayC={inC} overlayH={kH} overlayW={kW} overlayColor="#f9e2af"
            onClick={(r, c) => { const outR = Math.floor((r + padding[0]) / stride[0]); const outC = Math.floor((c + padding[1]) / stride[1]); onMove(Math.max(0, Math.min(outH - 1, outR)), Math.max(0, Math.min(outW - 1, outC))); }} />
        </div>
        <div className="tfm-conv-step-calc">
          <div className="tfm-conv-step-label">Kernel &times; Patch (ch {selectedInCh})</div>
          <CalcGrid values={patchValues} kH={kH} kW={kW} />
          <div className="tfm-conv-calc-result">
            {numInCh === 1 ? (<>sum = {fmtV(totalSum)}{bias !== 0 && ` + bias ${fmtV(bias)}`} = <strong>{fmtV(outputVal)}</strong></>) : (<>ch {selectedInCh} = {fmtV(currentChSum)}</>)}
          </div>
        </div>
        <div className="tfm-conv-step-panel">
          <div className="tfm-conv-step-label">Output (filter {selectedFilter})</div>
          <GridWithOverlay rawData={rawOutput} rows={outH} cols={outW} overlayR={posR} overlayC={posC} overlayH={1} overlayW={1} overlayColor="#89b4fa"
            onClick={(r, c) => onMove(Math.max(0, Math.min(outH - 1, r)), Math.max(0, Math.min(outW - 1, c)))} />
        </div>
      </div>
      {numInCh > 1 && <ChannelBreakdown perChannelSums={perChannelSums} selectedInCh={selectedInCh} onSelectInCh={onSelectInCh} totalSum={totalSum} bias={bias} outputVal={outputVal} />}
    </div>
  );
}

// ============================================================================
// ConvTranspose2d interactive — click INPUT pixel, see scattered output region
// ============================================================================

function TransposeConvInteractive({ rawInputs, allKernels, rawOutput, inH, inW, outH, outW, kH, kW, stride, padding,
  selectedFilter, selectedInCh, numInCh, posR, posC, onMove, onSelectInCh }: {
  rawInputs: number[][][][]; allKernels: number[][][][]; rawOutput: number[][];
  inH: number; inW: number; outH: number; outW: number;
  kH: number; kW: number; stride: number[]; padding: number[];
  selectedFilter: number; selectedInCh: number; numInCh: number;
  posR: number; posC: number;
  onMove: (r: number, c: number) => void; onSelectInCh: (ch: number) => void;
}) {
  // Clamp to input bounds
  const iR = Math.min(posR, Math.max(0, inH - 1));
  const iC = Math.min(posC, Math.max(0, inW - 1));

  const rawInput = rawInputs[selectedInCh] ?? rawInputs[0];
  const kernel = allKernels[selectedFilter]?.[selectedInCh] ?? allKernels[0]?.[0];
  const inputVal = rawInput[iR]?.[iC] ?? 0;

  // Where this input pixel scatters to in the output
  // ConvTranspose2d: output[iR*stride + kr - padding] += input[iR] * kernel[kr]
  const outStartR = iR * stride[0] - padding[0];
  const outStartC = iC * stride[1] - padding[1];

  // Kernel × input value = contributions to output
  const contributions: { kr: number; kc: number; kVal: number; outR: number; outC: number; contribution: number; inBounds: boolean }[] = [];
  for (let kr = 0; kr < kH; kr++) for (let kc = 0; kc < kW; kc++) {
    const oR = outStartR + kr;
    const oC = outStartC + kc;
    const kVal = kernel?.[kr]?.[kc] ?? 0;
    contributions.push({ kr, kc, kVal, outR: oR, outC: oC, contribution: inputVal * kVal, inBounds: oR >= 0 && oR < outH && oC >= 0 && oC < outW });
  }

  return (
    <div className="tfm-section">
      <div className="tfm-section-title">Step Through (Transpose) &middot; Filter {selectedFilter} &middot; input[{iR}, {iC}]</div>
      <div className="tfm-conv-formula">
        Each input pixel is multiplied by the kernel and <em>scattered</em> to a {kH}&times;{kW} output region
      </div>
      {numInCh > 1 && <ChannelSelector numInCh={numInCh} selectedInCh={selectedInCh} onSelectInCh={onSelectInCh} />}
      <div className="tfm-conv-step">
        <div className="tfm-conv-step-panel">
          <div className="tfm-conv-step-label">Input ch {selectedInCh}</div>
          <GridWithOverlay rawData={rawInput} rows={inH} cols={inW} overlayR={iR} overlayC={iC} overlayH={1} overlayW={1} overlayColor="#f9e2af"
            onClick={(r, c) => onMove(Math.max(0, Math.min(inH - 1, r)), Math.max(0, Math.min(inW - 1, c)))} />
          <div className="tfm-conv-calc-pos">value = {fmtV(inputVal)}</div>
        </div>
        <div className="tfm-conv-step-calc">
          <div className="tfm-conv-step-label">{fmtV(inputVal)} &times; Kernel</div>
          <div className="tfm-conv-calc-grid">
            {Array.from({ length: kH }, (_, kr) => (
              <div key={kr} className="tfm-conv-calc-row">
                {Array.from({ length: kW }, (_, kc) => {
                  const c = contributions[kr * kW + kc];
                  return (
                    <div key={kc} className={`tfm-conv-calc-cell ${!c.inBounds ? 'tfm-conv-calc-oob' : ''}`}>
                      <span className="tfm-conv-calc-k">{fmtV(c.kVal)}</span>
                      <span className="tfm-conv-calc-op">&rarr;</span>
                      <span className="tfm-conv-calc-in">{fmtV(c.contribution)}</span>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <div className="tfm-conv-calc-result">
            Scattered to output [{outStartR}:{outStartR + kH}, {outStartC}:{outStartC + kW}]
          </div>
        </div>
        <div className="tfm-conv-step-panel">
          <div className="tfm-conv-step-label">Output (filter {selectedFilter})</div>
          <GridWithOverlay rawData={rawOutput} rows={outH} cols={outW}
            overlayR={Math.max(0, outStartR)} overlayC={Math.max(0, outStartC)}
            overlayH={Math.min(kH, outH - Math.max(0, outStartR))} overlayW={Math.min(kW, outW - Math.max(0, outStartC))}
            overlayColor="#89b4fa"
            onClick={() => {}} />
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Shared sub-components
// ============================================================================

function ChannelSelector({ numInCh, selectedInCh, onSelectInCh }: { numInCh: number; selectedInCh: number; onSelectInCh: (ch: number) => void }) {
  return (
    <div className="tfm-conv-ch-selector">
      <span className="tfm-conv-ch-label">Input channel:</span>
      <div className="tfm-conv-ch-btns">
        {Array.from({ length: numInCh }, (_, i) => (
          <button key={i} className={`tfm-conv-ch-btn ${i === selectedInCh ? 'tfm-conv-ch-btn-active' : ''}`} onClick={() => onSelectInCh(i)}>{i}</button>
        ))}
      </div>
    </div>
  );
}

function ChannelBreakdown({ perChannelSums, selectedInCh, onSelectInCh, totalSum, bias, outputVal }: {
  perChannelSums: number[]; selectedInCh: number; onSelectInCh: (ch: number) => void; totalSum: number; bias: number; outputVal: number;
}) {
  return (
    <div className="tfm-conv-breakdown">
      <div className="tfm-conv-breakdown-items">
        {perChannelSums.map((s, i) => (
          <span key={i} className={`tfm-conv-breakdown-item ${i === selectedInCh ? 'tfm-conv-breakdown-active' : ''}`} onClick={() => onSelectInCh(i)}>ch{i}: {fmtV(s)}</span>
        ))}
      </div>
      <div className="tfm-conv-breakdown-total">= {fmtV(totalSum)}{bias !== 0 ? ` + bias ${fmtV(bias)}` : ''} = <strong>{fmtV(outputVal)}</strong></div>
    </div>
  );
}

function CalcGrid({ values, kH, kW }: { values: { inVal: number; kVal: number }[]; kH: number; kW: number }) {
  return (
    <div className="tfm-conv-calc-grid">
      {Array.from({ length: kH }, (_, kr) => (
        <div key={kr} className="tfm-conv-calc-row">
          {Array.from({ length: kW }, (_, kc) => {
            const p = values[kr * kW + kc];
            return (
              <div key={kc} className="tfm-conv-calc-cell">
                <span className="tfm-conv-calc-in">{fmtV(p.inVal)}</span>
                <span className="tfm-conv-calc-op">&times;</span>
                <span className="tfm-conv-calc-k">{fmtV(p.kVal)}</span>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function GridWithOverlay({ rawData, rows, cols, overlayR, overlayC, overlayH, overlayW, overlayColor, onClick }: {
  rawData: number[][]; rows: number; cols: number;
  overlayR: number; overlayC: number; overlayH: number; overlayW: number;
  overlayColor: string; onClick: (r: number, c: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cellSize = Math.max(4, Math.min(12, Math.floor(220 / Math.max(rows, cols))));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = cols * cellSize, h = rows * cellSize;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const vals = rawData.flat();
    const vmin = Math.min(...vals), vmax = Math.max(...vals);
    const vrange = vmax - vmin || 1;
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const v = Math.round((((rawData[r]?.[c] ?? 0) - vmin) / vrange) * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
    }
    // Dim outside
    if (overlayH > 1 || overlayW > 1) {
      const ox = overlayC * cellSize, oy = overlayR * cellSize, ow = overlayW * cellSize, oh = overlayH * cellSize;
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      if (oy > 0) ctx.fillRect(0, 0, w, Math.max(0, oy));
      const botY = oy + oh;
      if (botY < h) ctx.fillRect(0, botY, w, h - botY);
      if (ox > 0) ctx.fillRect(0, Math.max(0, oy), Math.max(0, ox), oh);
      const rightX = ox + ow;
      if (rightX < w) ctx.fillRect(rightX, Math.max(0, oy), w - rightX, oh);
    }
    ctx.strokeStyle = overlayColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(overlayC * cellSize + 1, overlayR * cellSize + 1, overlayW * cellSize - 2, overlayH * cellSize - 2);
  }, [rawData, rows, cols, overlayR, overlayC, overlayH, overlayW, overlayColor, cellSize]);

  return <canvas ref={canvasRef} className="tfm-conv-step-canvas" style={{ width: cols * cellSize, height: rows * cellSize, cursor: 'crosshair' }}
    onClick={(e) => { const rect = canvasRef.current!.getBoundingClientRect(); onClick(Math.max(0, Math.min(rows - 1, Math.floor((e.clientY - rect.top) / rect.height * rows))), Math.max(0, Math.min(cols - 1, Math.floor((e.clientX - rect.left) / rect.width * cols)))); }} />;
}

function fmtV(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(2);
  if (abs >= 0.01) return v.toFixed(3);
  return v.toExponential(1);
}
