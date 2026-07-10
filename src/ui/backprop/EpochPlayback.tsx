// EpochPlayback — the main backprop view. Backprop's whole job is to make the
// predictions better; this lets you watch that happen. It replays the fixed
// "tracked" examples recorded each epoch during the last training run, with a
// scrubber (play / step / rewind) so you can fast-forward and rewind through
// training and see how each example's prediction changes.

import { useEffect, useState } from 'react';
import type { EpochData, TrackedSampleProbe } from '../dashboard/types';
import { PixelCanvas } from '../PixelCanvas';

export function EpochPlayback({ progress }: { progress: EpochData[] }) {
  const hasData = progress.length > 0 && !!progress[0].trackedSamples?.length;
  const last = Math.max(0, progress.length - 1);
  const [rawIdx, setEpochIdx] = useState(last); // page remounts fresh on each open
  const epochIdx = Math.min(rawIdx, last); // stay in range if the run changes while open
  const [playing, setPlaying] = useState(false);

  // Auto-advance while playing. Both setStates live in the timeout callback
  // (async), and we stop once the next tick reaches the final epoch.
  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      setEpochIdx((i) => Math.min(last, i + 1));
      if (epochIdx + 1 >= last) setPlaying(false);
    }, 650);
    return () => clearTimeout(t);
  }, [playing, epochIdx, last]);

  if (!hasData) {
    return (
      <div className="backprop-page bp-play-empty">
        <div className="bp-play-empty-inner">
          <div className="bp-play-empty-title">Watch the model learn</div>
          <div className="bp-play-empty-text">
            Backprop's whole job is to make the predictions better. Train the model (the{' '}
            <strong>Train</strong> button) and this will replay how its predictions on a few fixed
            examples change, epoch by epoch — you can scrub forward and back through training.
          </div>
        </div>
      </div>
    );
  }

  const firstProbes = progress[0].trackedSamples!; // images live on epoch 0
  const epoch = progress[epochIdx];
  const probes = epoch.trackedSamples ?? [];
  const isClassification = firstProbes.some((p) => p.probabilities?.length);
  const classNames = progress[0].classNames ?? null;
  const clsLabel = (c: number | null | undefined): string =>
    c == null ? '?' : classNames && c < classNames.length ? classNames[c] : String(c);

  const play = () => {
    if (epochIdx >= last) setEpochIdx(0); // replay from the start
    setPlaying((p) => !p);
  };

  return (
    <div className="backprop-page">
      <div className="bp-play-head">
        <div className="bp-play-title">Watch the predictions improve</div>
        <div className="bp-play-metrics">
          <span>
            epoch <strong>{epoch.epoch}</strong> / {progress[last].epoch}
          </span>
          <span>
            loss <strong>{epoch.loss?.toFixed(4)}</strong>
          </span>
          {epoch.accuracy != null && (
            <span>
              acc <strong>{(epoch.accuracy * 100).toFixed(1)}%</strong>
            </span>
          )}
        </div>
      </div>

      <div className="bp-play-samples">
        {firstProbes.map((s, i) => (
          <SampleCard
            key={s.idx}
            image={s}
            probe={probes[i]}
            classification={isClassification}
            clsLabel={clsLabel}
          />
        ))}
      </div>

      <TrainingCurve progress={progress} epochIdx={epochIdx} onSeek={setEpochIdx} />

      <div className="bp-play-controls">
        <button
          className="step-through-ctrl"
          onClick={() => setEpochIdx(0)}
          disabled={epochIdx === 0}
        >
          &#x23EE;
        </button>
        <button
          className="step-through-ctrl"
          onClick={() => setEpochIdx((i) => Math.max(0, i - 1))}
          disabled={epochIdx === 0}
        >
          &#x25C0;
        </button>
        <button className="step-through-ctrl bp-play-btn" onClick={play}>
          {playing ? '⏸' : '▶'}
        </button>
        <button
          className="step-through-ctrl"
          onClick={() => setEpochIdx((i) => Math.min(last, i + 1))}
          disabled={epochIdx >= last}
        >
          &#x25B6;
        </button>
        <button
          className="step-through-ctrl"
          onClick={() => setEpochIdx(last)}
          disabled={epochIdx >= last}
        >
          &#x23ED;
        </button>
        <input
          type="range"
          min={0}
          max={last}
          value={epochIdx}
          onChange={(e) => {
            setPlaying(false);
            setEpochIdx(parseInt(e.target.value, 10));
          }}
          className="step-through-slider"
        />
        <span className="step-through-counter">
          epoch {epoch.epoch} / {progress[last].epoch}
        </span>
      </div>
    </div>
  );
}

/** One example: its image, true label, and its prediction at the current epoch. */
function SampleCard({
  image,
  probe,
  classification,
  clsLabel,
}: {
  image: TrackedSampleProbe;
  probe: TrackedSampleProbe | undefined;
  classification: boolean;
  clsLabel: (c: number | null | undefined) => string;
}) {
  const trueLabel = image.label;
  const pred = probe?.predictedClass;
  const correct = pred != null && pred === trueLabel;
  const probs = probe?.probabilities;

  // Which classes to show as bars: top few + the true class.
  let rows: number[] = [];
  if (probs) {
    const idx = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
    const set = new Set(idx.slice(0, 5));
    if (trueLabel != null) set.add(trueLabel);
    rows = [...set].sort((a, b) => (probs[b] ?? 0) - (probs[a] ?? 0));
  }

  return (
    <div className="bp-play-card">
      <div className="bp-play-card-top">
        <SampleImage image={image} />
        <div className="bp-play-card-meta">
          <div className="bp-play-truth">
            true <strong>{clsLabel(trueLabel)}</strong>
          </div>
          {classification ? (
            pred != null ? (
              <div className={`bp-play-pred ${correct ? 'ok' : 'wrong'}`}>
                pred <strong>{clsLabel(pred)}</strong> {correct ? '✓' : '✗'}
              </div>
            ) : (
              <div className="bp-play-pred">—</div>
            )
          ) : (
            <div className="bp-play-pred">loss {probe?.loss?.toFixed(3) ?? '—'}</div>
          )}
        </div>
      </div>

      {classification && probs && (
        <div className="bp-play-bars">
          {rows.map((c) => {
            const p = probs[c] ?? 0;
            const isTrue = c === trueLabel;
            const isPred = c === pred;
            return (
              <div key={c} className="bp-play-bar-row">
                <span className={`bp-play-bar-lab ${isTrue ? 'true' : ''}`} title={clsLabel(c)}>
                  {clsLabel(c)}
                </span>
                <div className="bp-play-bar-track">
                  <div
                    className={`bp-play-bar-fill ${isTrue ? 'true' : isPred ? 'pred' : ''}`}
                    style={{ width: `${p * 100}%` }}
                  />
                </div>
                <span className="bp-play-bar-pct">{(p * 100).toFixed(0)}%</span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function SampleImage({ image }: { image: TrackedSampleProbe }) {
  if (!image.imagePixels) return <div className="bp-play-noimg">#{image.idx}</div>;
  return <PixelCanvas pixels={image.imagePixels} className="bp-play-img" />;
}

/** Training curve — loss (falling) and accuracy (rising) across epochs, with a
 * marker at the current epoch. Click to seek. */
function TrainingCurve({
  progress,
  epochIdx,
  onSeek,
}: {
  progress: EpochData[];
  epochIdx: number;
  onSeek: (i: number) => void;
}) {
  const n = progress.length;
  const losses = progress.map((e) => e.loss ?? 0);
  const accs = progress.map((e) => e.accuracy ?? null);
  const hasAcc = accs.some((a) => a != null);
  const maxLoss = Math.max(...losses, 1e-9);

  const W = 900;
  const H = 160;
  const pad = { l: 44, r: hasAcc ? 46 : 14, t: 22, b: 24 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;
  const xOf = (i: number) => pad.l + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);
  const yLoss = (l: number) => pad.t + ph * (1 - l / maxLoss);
  const yAcc = (a: number) => pad.t + ph * (1 - a);
  const lossPath = losses
    .map((l, i) => `${i === 0 ? 'M' : 'L'} ${xOf(i).toFixed(1)} ${yLoss(l).toFixed(1)}`)
    .join(' ');
  const lossArea = `${lossPath} L ${xOf(n - 1).toFixed(1)} ${pad.t + ph} L ${xOf(0).toFixed(1)} ${pad.t + ph} Z`;
  const accPath = accs
    .map((a, i) =>
      a == null ? null : `${i === 0 ? 'M' : 'L'} ${xOf(i).toFixed(1)} ${yAcc(a).toFixed(1)}`,
    )
    .filter(Boolean)
    .join(' ');

  const seek = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = (((e.clientX - rect.left) / rect.width) * W - pad.l) / pw;
    onSeek(Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1)))));
  };

  return (
    <div className="bp-play-curve-wrap">
      <svg className="bp-play-curve" viewBox={`0 0 ${W} ${H}`} onClick={seek}>
        <line x1={pad.l} y1={pad.t + ph} x2={W - pad.r} y2={pad.t + ph} stroke="#313244" />
        <path d={lossArea} fill="rgba(249, 226, 175, 0.10)" />
        <path d={lossPath} fill="none" stroke="#f9e2af" strokeWidth="2" />
        {hasAcc && <path d={accPath} fill="none" stroke="#a6e3a1" strokeWidth="2" />}

        <line
          x1={xOf(epochIdx)}
          y1={pad.t}
          x2={xOf(epochIdx)}
          y2={pad.t + ph}
          stroke="#cdd6f4"
          strokeWidth="1"
          strokeDasharray="3 3"
        />
        <circle cx={xOf(epochIdx)} cy={yLoss(losses[epochIdx])} r="4" fill="#f9e2af" />
        {hasAcc && accs[epochIdx] != null && (
          <circle cx={xOf(epochIdx)} cy={yAcc(accs[epochIdx]!)} r="4" fill="#a6e3a1" />
        )}

        {/* legend + axes */}
        <text x={pad.l} y={13} fill="#f9e2af" fontSize="12">
          loss
        </text>
        {hasAcc && (
          <text x={pad.l + 44} y={13} fill="#a6e3a1" fontSize="12">
            accuracy
          </text>
        )}
        <text x={pad.l - 6} y={pad.t + 5} fill="#6c7086" fontSize="11" textAnchor="end">
          {maxLoss.toFixed(2)}
        </text>
        <text x={pad.l - 6} y={pad.t + ph} fill="#6c7086" fontSize="11" textAnchor="end">
          0
        </text>
        {hasAcc && (
          <>
            <text x={W - pad.r + 6} y={pad.t + 5} fill="#6c7086" fontSize="11">
              100%
            </text>
            <text x={W - pad.r + 6} y={pad.t + ph} fill="#6c7086" fontSize="11">
              0%
            </text>
          </>
        )}
      </svg>
    </div>
  );
}
