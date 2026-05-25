// AttentionViz — per-head heatmap + focused-row drill-in for attention weights.
//
// Compact inline view:
//   [ Head tabs (with focus badge) | ? help | Expand ⛶ ]
//   [ heatmap canvas with axis labels       ]
//   [ legend bar ]
//   [ focused query row: bar chart with token labels ]
//
// Expanded modal view: same content, but heatmap fills the viewport.

import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import type { AttentionTransformation, AttentionFocusRow } from '../types';

const MIN_CANVAS = 96;
const INLINE_MAX_CANVAS = 480;
const MAX_TOKEN_LABEL_LEN = 8;   // truncate token strings beyond this with "…"

export function AttentionViz({ t }: { t: AttentionTransformation }) {
  const [selected, setSelected] = useState<number>(0);
  const [focusIdx, setFocusIdx] = useState<number>(t.focusRow?.queryIndex ?? t.displaySize - 1);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [helpOpen, setHelpOpen] = useState<boolean>(false);

  // Esc closes the expanded modal
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.stopPropagation(); setExpanded(false); }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [expanded]);

  const matrix = t.perHeadWeights[selected] ?? [];
  const downsampled = t.seqLen !== t.displaySize;

  const controls = (
    <div className="tfm-attention-controls">
      {t.perHeadWeights.map((_, i) => {
        const ent = t.headEntropy?.[i];
        const entLabel = ent != null ? entropyLabel(ent) : null;
        return (
          <button
            key={i}
            className={`tfm-attn-tab ${selected === i ? 'tfm-attn-tab-active' : ''}`}
            onClick={() => setSelected(i)}
            title={ent != null ? `Head ${i + 1} — focus score ${ent.toFixed(2)} (${entLabel?.label})` : `Head ${i + 1}`}
          >
            Head {i + 1}
            {entLabel != null && (
              <span className={`tfm-attn-entropy tfm-attn-entropy-${entLabel.cls}`}>{entLabel.label}</span>
            )}
          </button>
        );
      })}
      <button
        className="tfm-attn-help"
        onClick={() => setHelpOpen((o) => !o)}
        title="What does focus mean?"
      >?</button>

      <div className="tfm-attn-meta">
        {t.numHeads} head{t.numHeads > 1 ? 's' : ''} · {t.seqLen} pos
        {t.causalMask ? ' · causal' : ''}
        {downsampled ? ` · heatmap ${t.displaySize}×${t.displaySize}` : ''}
      </div>

      {!expanded && (
        <button className="tfm-attn-expand" onClick={() => setExpanded(true)} title="Expand to full screen">
          ⛶ Expand
        </button>
      )}
    </div>
  );

  const helpBlurb = helpOpen && (
    <div className="tfm-attn-help-blurb">
      <b>Focus score</b> = how concentrated each head's attention is, averaged across query positions.
      <ul>
        <li><span className="tfm-attn-help-pill tfm-attn-entropy-sharp">focused</span> — attends to a few positions (often nearby tokens or specific syntactic roles)</li>
        <li><span className="tfm-attn-help-pill tfm-attn-entropy-mid">mixed</span> — combines a handful of positions</li>
        <li><span className="tfm-attn-help-pill tfm-attn-entropy-diffuse">diffuse</span> — spreads attention broadly, close to uniform</li>
      </ul>
      Computed as normalized Shannon entropy of each row (0 = single position, 1 = uniform).
    </div>
  );

  // Inline view: compact heatmap only — drill-in lives in the fullscreen modal.
  if (!expanded) {
    return (
      <div className="tfm-attention">
        {controls}
        {helpBlurb}
        <div className="tfm-attention-body">
          <AttentionHeatmap
            matrix={matrix}
            tokens={t.tokens}
            focusIdx={focusIdx}
            onPickRow={(idx) => { setFocusIdx(idx); setExpanded(true); }}
            maxCanvas={INLINE_MAX_CANVAS}
          />
          <AttentionLegend />
        </div>
      </div>
    );
  }

  // Expanded modal view — heatmap + focused-row inside a fullscreen overlay
  return (
    <>
      <div className="tfm-attention">
        {controls}
        {helpBlurb}
        <div className="tfm-attention-collapsed-note">Open in fullscreen ↗</div>
      </div>
      <div className="tfm-attention-modal-overlay" onClick={() => setExpanded(false)}>
        <div className="tfm-attention-modal" onClick={(e) => e.stopPropagation()}>
          <div className="tfm-attention-modal-header">
            <span className="tfm-attention-modal-title">
              Attention · Head {selected + 1}
              {t.headEntropy?.[selected] != null && (
                <span className={`tfm-attn-entropy tfm-attn-entropy-${entropyLabel(t.headEntropy[selected]).cls}`}>
                  {entropyLabel(t.headEntropy[selected]).label}
                </span>
              )}
            </span>
            <div className="tfm-attention-modal-tabs">
              {t.perHeadWeights.map((_, i) => (
                <button
                  key={i}
                  className={`tfm-attn-tab ${selected === i ? 'tfm-attn-tab-active' : ''}`}
                  onClick={() => setSelected(i)}
                >
                  Head {i + 1}
                </button>
              ))}
            </div>
            <button className="tfm-attention-modal-close" onClick={() => setExpanded(false)} title="Close (Esc)">×</button>
          </div>

          <div className="tfm-attention-modal-body">
            <div className="tfm-attention-modal-heatmap">
              <ExpandedHeatmap
                matrix={matrix}
                tokens={t.tokens}
                focusIdx={focusIdx}
                onPickRow={setFocusIdx}
              />
            </div>
            <div className="tfm-attention-modal-footer">
              <AttentionLegend />
              {t.focusRow && (
                <FocusedRowView
                  focus={t.focusRow}
                  selectedHead={selected}
                  seqLen={t.seqLen}
                  displaySize={t.displaySize}
                  clickedDisplayRow={focusIdx}
                  allWeights={matrix}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ---- Inline heatmap (fixed-ish size, hover tooltip, click-to-focus) ----

interface HeatmapProps {
  matrix: number[][];
  tokens?: string[];
  focusIdx: number;
  onPickRow: (idx: number) => void;
  maxCanvas: number;
}

function AttentionHeatmap({ matrix, tokens, focusIdx, onPickRow, maxCanvas }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = matrix.length;
  const [hover, setHover] = useState<{ r: number; c: number; v: number; x: number; y: number } | null>(null);

  const cell = useMemo(() => {
    if (size === 0) return 4;
    const ideal = Math.floor(maxCanvas / size);
    return Math.max(1, Math.min(8, ideal));
  }, [size, maxCanvas]);

  const canvasSize = Math.max(MIN_CANVAS, size * cell);
  const showLabels = !!tokens && tokens.length === size && size <= 64;

  const globalMax = useMemo(() => {
    let m = 0;
    for (const row of matrix) for (const v of row) if (v > m) m = v;
    return m > 0 ? m : 1;
  }, [matrix]);

  useEffect(() => paintHeatmap(canvasRef.current, matrix, cell, size, globalMax, focusIdx),
    [matrix, cell, size, globalMax, focusIdx]);

  const onMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.floor((x / rect.width) * size);
    const r = Math.floor((y / rect.height) * size);
    if (r < 0 || r >= size || c < 0 || c >= size) { setHover(null); return; }
    setHover({ r, c, v: matrix[r][c], x, y });
  }, [matrix, size]);

  const onLeave = () => setHover(null);
  const onClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const r = Math.floor(((e.clientY - rect.top) / rect.height) * size);
    if (r >= 0 && r < size) onPickRow(r);
  }, [size, onPickRow]);

  const renderScale = canvasSize / (size * cell);

  return (
    <div className="tfm-attention-canvas-wrap">
      <div className="tfm-attention-axis-label tfm-attention-axis-y">Query →</div>
      <div className="tfm-attention-canvas-stack">
        <canvas
          ref={canvasRef}
          className="tfm-attention-canvas"
          style={{ width: canvasSize, height: canvasSize, imageRendering: 'pixelated', cursor: 'crosshair' }}
          onMouseMove={onMove}
          onMouseLeave={onLeave}
          onClick={onClick}
        />
        {showLabels && tokens && <TokenAxes tokens={tokens} cell={cell * renderScale} canvasSize={canvasSize} />}
        {hover && <HoverTooltip hover={hover} tokens={tokens} canvasSize={canvasSize} />}
      </div>
      <div className="tfm-attention-axis-label tfm-attention-axis-x">Key →</div>
      <div className="tfm-attention-hint">Click a row to focus that query position</div>
    </div>
  );
}

// ---- Expanded heatmap (fills modal, scrollable when very large) ----

function ExpandedHeatmap({ matrix, tokens, focusIdx, onPickRow }: Omit<HeatmapProps, 'maxCanvas'>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const size = matrix.length;
  const [hover, setHover] = useState<{ r: number; c: number; v: number; x: number; y: number } | null>(null);
  const [viewport, setViewport] = useState({ w: 800, h: 600 });

  useEffect(() => {
    function measure() {
      if (!wrapRef.current) return;
      const r = wrapRef.current.getBoundingClientRect();
      setViewport({ w: Math.floor(r.width), h: Math.floor(r.height) });
    }
    measure();
    window.addEventListener('resize', measure);
    // ResizeObserver picks up layout changes that 'resize' doesn't (e.g. flex re-flow)
    let ro: ResizeObserver | null = null;
    if (wrapRef.current && typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(measure);
      ro.observe(wrapRef.current);
    }
    return () => {
      window.removeEventListener('resize', measure);
      if (ro) ro.disconnect();
    };
  }, []);

  // Fit cell size so the heatmap fills as much of the modal as possible — use
  // the SMALLER axis to keep the matrix square, but allow it to fill ~90% of
  // both dimensions instead of half.
  const cell = useMemo(() => {
    if (size === 0) return 4;
    const padding = 80; // axis labels + margin
    const fit = Math.floor(Math.min(viewport.w - padding, viewport.h - padding) / size);
    return Math.max(1, fit);
  }, [size, viewport]);

  const canvasSize = size * cell;
  // Token labels only fit when cells are tall/wide enough to show characters.
  const showLabels = !!tokens && tokens.length === size && cell >= 12;

  const globalMax = useMemo(() => {
    let m = 0;
    for (const row of matrix) for (const v of row) if (v > m) m = v;
    return m > 0 ? m : 1;
  }, [matrix]);

  useEffect(() => paintHeatmap(canvasRef.current, matrix, cell, size, globalMax, focusIdx),
    [matrix, cell, size, globalMax, focusIdx]);

  const onMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.floor((x / rect.width) * size);
    const r = Math.floor((y / rect.height) * size);
    if (r < 0 || r >= size || c < 0 || c >= size) { setHover(null); return; }
    setHover({ r, c, v: matrix[r][c], x, y });
  }, [matrix, size]);

  const onClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current; if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const r = Math.floor(((e.clientY - rect.top) / rect.height) * size);
    if (r >= 0 && r < size) onPickRow(r);
  }, [size, onPickRow]);

  return (
    <div className="tfm-attention-expanded-wrap" ref={wrapRef}>
      <div className="tfm-attention-canvas-wrap tfm-attention-canvas-wrap-expanded">
        <div className="tfm-attention-axis-label tfm-attention-axis-y">Query →</div>
        <div className="tfm-attention-canvas-stack">
          <canvas
            ref={canvasRef}
            className="tfm-attention-canvas"
            style={{ width: canvasSize, height: canvasSize, imageRendering: 'pixelated', cursor: 'crosshair' }}
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
            onClick={onClick}
          />
          {showLabels && tokens && <TokenAxes tokens={tokens} cell={cell} canvasSize={canvasSize} />}
          {hover && <HoverTooltip hover={hover} tokens={tokens} canvasSize={canvasSize} />}
        </div>
        <div className="tfm-attention-axis-label tfm-attention-axis-x">Key →</div>
      </div>
    </div>
  );
}

// ---- Shared rendering helpers ----

function paintHeatmap(
  canvas: HTMLCanvasElement | null,
  matrix: number[][],
  cell: number,
  size: number,
  globalMax: number,
  focusIdx: number,
) {
  if (!canvas || size === 0) return;
  canvas.width = size * cell;
  canvas.height = size * cell;
  const ctx = canvas.getContext('2d'); if (!ctx) return;

  const img = ctx.createImageData(size * cell, size * cell);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = matrix[r][c] / globalMax;
      const [R, G, B] = heatColor(v);
      for (let dy = 0; dy < cell; dy++) {
        for (let dx = 0; dx < cell; dx++) {
          const x = c * cell + dx;
          const y = r * cell + dy;
          const idx = (y * size * cell + x) * 4;
          img.data[idx] = R;
          img.data[idx + 1] = G;
          img.data[idx + 2] = B;
          img.data[idx + 3] = 255;
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  if (focusIdx >= 0 && focusIdx < size) {
    ctx.strokeStyle = 'rgba(249, 226, 175, 0.85)';
    ctx.lineWidth = Math.max(1, Math.floor(cell / 2));
    ctx.strokeRect(0, focusIdx * cell, size * cell, cell);
  }
}

function TokenAxes({ tokens, cell, canvasSize }: { tokens: string[]; cell: number; canvasSize: number }) {
  return (
    <>
      <div className="tfm-attention-tokens tfm-attention-tokens-x" style={{ width: canvasSize }}>
        {tokens.map((tok, i) => (
          <span
            key={i}
            className="tfm-attention-token-x"
            style={{ width: cell }}
            title={`pos ${i}: ${JSON.stringify(tok)}`}
          >
            {displayToken(tok)}
          </span>
        ))}
      </div>
      <div className="tfm-attention-tokens tfm-attention-tokens-y" style={{ height: canvasSize }}>
        {tokens.map((tok, i) => (
          <span
            key={i}
            className="tfm-attention-token-y"
            style={{ height: cell }}
            title={`pos ${i}: ${JSON.stringify(tok)}`}
          >
            {displayToken(tok)}
          </span>
        ))}
      </div>
    </>
  );
}

function HoverTooltip({ hover, tokens, canvasSize }: {
  hover: { r: number; c: number; v: number; x: number; y: number };
  tokens?: string[];
  canvasSize: number;
}) {
  return (
    <div
      className="tfm-attention-tooltip"
      style={{ left: Math.min(hover.x + 12, canvasSize - 180), top: Math.max(hover.y - 48, 0) }}
    >
      <div>q={hover.r}{tokens?.[hover.r] != null ? ` · ${displayTokenInline(tokens[hover.r])}` : ''}</div>
      <div>k={hover.c}{tokens?.[hover.c] != null ? ` · ${displayTokenInline(tokens[hover.c])}` : ''}</div>
      <div className="tfm-attention-tooltip-w">{(hover.v * 100).toFixed(1)}%</div>
    </div>
  );
}

function AttentionLegend() {
  const stops = useMemo(() => {
    const n = 32;
    return Array.from({ length: n }, (_, i) => {
      const v = i / (n - 1);
      const [r, g, b] = heatColor(v);
      return `rgb(${r}, ${g}, ${b}) ${(v * 100).toFixed(0)}%`;
    }).join(', ');
  }, []);
  return (
    <div className="tfm-attention-legend">
      <div className="tfm-attention-legend-label">Low</div>
      <div className="tfm-attention-legend-bar" style={{ background: `linear-gradient(to right, ${stops})` }} />
      <div className="tfm-attention-legend-label">High</div>
    </div>
  );
}

// ---- Focused query row: bar chart + top-K tokens ----

interface FocusedRowProps {
  focus: AttentionFocusRow;
  selectedHead: number;
  seqLen: number;          // true sequence length
  displaySize: number;     // heatmap row count
  clickedDisplayRow: number;
  allWeights: number[][];  // [head's heatmap matrix at displaySize×displaySize]
}

function FocusedRowView({ focus, selectedHead, seqLen, displaySize, clickedDisplayRow, allWeights }: FocusedRowProps) {
  // Pick the right data source for the clicked row:
  //   - if it's the default focus row (last position by default), the backend ships full-resolution per-head data
  //   - otherwise, use the heatmap matrix row. When seqLen === displaySize, that row is already full-resolution.
  const isDefaultFocus = clickedDisplayRow === focus.queryIndex;
  const noDownsampling = seqLen === displaySize;
  const fullResolution = isDefaultFocus || noDownsampling;

  let row: number[];
  let labels: string[] | null;
  let top: { index: number; weight: number }[];

  if (isDefaultFocus) {
    row = focus.perHeadRow[selectedHead] ?? focus.avgRow;
    labels = focus.labels;
    top = focus.perHeadTop[selectedHead] ?? focus.avgTop;
  } else {
    row = allWeights[clickedDisplayRow] ?? [];
    // When seqLen matches displaySize, focus.labels indices map 1:1 to row indices.
    labels = noDownsampling ? focus.labels : null;
    top = topKFromRow(row, 5);
  }

  const maxVal = Math.max(0.0001, ...row);
  const clickedToken = labels && labels[clickedDisplayRow] != null ? labels[clickedDisplayRow] : null;
  const queryLabel = isDefaultFocus && focus.queryToken != null
    ? `${focus.queryIndex}: ${displayTokenInline(focus.queryToken)}`
    : clickedToken != null
      ? `${clickedDisplayRow}: ${displayTokenInline(clickedToken)}`
      : `${clickedDisplayRow}`;

  return (
    <div className="tfm-attention-focus">
      <div className="tfm-attention-focus-header">
        <div className="tfm-attention-focus-title">
          Query position <b>{queryLabel}</b>
          {!fullResolution && (
            <span className="tfm-attention-focus-note"> · showing downsampled row ({row.length} cells)</span>
          )}
        </div>
        <div className="tfm-attention-focus-top">
          <span className="tfm-attention-focus-top-label">Top attended:</span>
          {top.map((t, i) => (
            <span key={i} className="tfm-attention-focus-top-chip" title={`weight ${(t.weight * 100).toFixed(2)}%`}>
              {labels && labels[t.index] != null
                ? `${t.index}:${displayToken(labels[t.index])}`
                : `pos ${t.index}`}
              <span className="tfm-attention-focus-top-w">{(t.weight * 100).toFixed(1)}%</span>
            </span>
          ))}
        </div>
      </div>

      <div className="tfm-attention-focus-bars">
        {row.map((v, i) => {
          const isTop = top.some((tk) => tk.index === i);
          const h = Math.max(1, Math.round((v / maxVal) * 100));
          return (
            <div
              key={i}
              className={`tfm-attention-focus-bar ${isTop ? 'tfm-attention-focus-bar-top' : ''}`}
              style={{ height: `${h}%` }}
              title={`pos ${i}${labels?.[i] != null ? ` (${displayTokenInline(labels[i])})` : ''} · ${(v * 100).toFixed(2)}%`}
            />
          );
        })}
      </div>

      {labels && (
        <div className="tfm-attention-focus-labels">
          {labels.map((tok, i) => (
            <span key={i} className="tfm-attention-focus-label" title={`pos ${i}: ${JSON.stringify(tok)}`}>
              {displayToken(tok)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ---- Helpers ----

function topKFromRow(row: number[], k: number): { index: number; weight: number }[] {
  return row.map((w, i) => ({ index: i, weight: w }))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, k);
}

function entropyLabel(e: number): { label: string; cls: 'sharp' | 'mid' | 'diffuse' } {
  if (e < 0.4) return { label: 'focused', cls: 'sharp' };
  if (e < 0.75) return { label: 'mixed', cls: 'mid' };
  return { label: 'diffuse', cls: 'diffuse' };
}

// Compact label for tight spaces — truncates and reveals whitespace.
function displayToken(t: string): string {
  if (t === '\n') return '↵';
  if (t === ' ') return '␣';
  if (t === '\t') return '⇥';
  if (t === '') return '∅';
  if (t.length > MAX_TOKEN_LABEL_LEN) return t.slice(0, MAX_TOKEN_LABEL_LEN - 1) + '…';
  return t;
}

// Used in tooltips/headers — shows whitespace marker but keeps full text.
function displayTokenInline(t: string): string {
  if (t === '\n') return '"↵"';
  if (t === ' ') return '"␣"';
  if (t === '\t') return '"⇥"';
  if (t === '') return '"(empty)"';
  return JSON.stringify(t);
}

// "viridis-ish" gradient — dark purple → blue → green → yellow
function heatColor(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  if (x < 0.25) {
    const t = x / 0.25;
    return [Math.round(30 + t * 20), 30, Math.round(60 + t * 80)];
  } else if (x < 0.5) {
    const t = (x - 0.25) / 0.25;
    return [Math.round(50 + t * -30), Math.round(30 + t * 110), Math.round(140 + t * 20)];
  } else if (x < 0.75) {
    const t = (x - 0.5) / 0.25;
    return [Math.round(20 + t * 120), Math.round(140 + t * 60), Math.round(160 + t * -100)];
  } else {
    const t = (x - 0.75) / 0.25;
    return [Math.round(140 + t * 115), Math.round(200 + t * 40), Math.round(60 + t * -20)];
  }
}
