// PerLayerView — the "per layer" backprop page (a drill-in from the main epoch
// playback). Steps right → left through the backward pass: the loss seed, then
// each layer's cell (grad in → the update → grad out), plus one-step before/after
// and wiggle-a-weight. Self-contained so the page is easy to drop.

import { useEffect, useState, useCallback } from 'react';
import type {
  BackwardResult,
  BackwardStage,
  BackwardSample,
  OneStepResult,
  StepLayer,
} from './types';
import type { FeatureMaps } from '../step-through/types';
import { StageTimeline } from '../step-through/StageTimeline';
import { BackpropDetail } from './BackpropDetail';
import { OneStepBanner } from './OneStepBanner';
import { apiUrl } from '../../api/base';

/** Grayscale thumbnail (FeatureMaps) from the sample's pixels, for the data card. */
function sampleToFeatureMaps(
  pixels: number[][] | number[][][],
  channels: number,
): FeatureMaps | undefined {
  if (!pixels || pixels.length === 0) return undefined;
  const h = pixels.length;
  const rgb = Array.isArray((pixels[0] as number[][])[0]);
  const map: number[][] = rgb
    ? (pixels as number[][][]).map((row) => row.map((px) => (px[0] + px[1] + px[2]) / 3))
    : (pixels as number[][]);
  const w = map[0]?.length ?? 0;
  return { maps: [map], channels, showing: 1, height: h, width: w };
}

/** A synthetic "data" stage at the left of the backward timeline — the input the
 * gradients were computed for. Backprop skips data nodes, so we rebuild one. */
function buildDataStage(sample: BackwardSample): BackwardStage {
  const fmaps = sample.imagePixels
    ? sampleToFeatureMaps(sample.imagePixels, sample.imageChannels ?? 1)
    : undefined;
  return {
    stageId: 'bwd/__data__',
    path: ['__data__'],
    nodeId: '__data__',
    nodeType: sample.datasetType ?? 'data',
    displayName: 'Data',
    depth: 0,
    outputShape: fmaps ? [fmaps.channels, fmaps.height, fmaps.width] : undefined,
    transformation: fmaps ? { type: 'data', featureMaps: fmaps } : undefined,
  };
}

export function PerLayerView({ graphJson }: { graphJson: string }) {
  const [stages, setStages] = useState<BackwardStage[]>([]);
  const [loss, setLoss] = useState<number | null>(null);
  const [sample, setSample] = useState<BackwardSample | null>(null);
  const [sampleIdx, setSampleIdx] = useState<number | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  // One-step (before/after) state — computed lazily on demand.
  const [step, setStep] = useState<OneStepResult | null>(null);
  const [showAfter, setShowAfter] = useState(false); // Before ⟷ After toggle
  const [stepLoading, setStepLoading] = useState(false);
  const [stepError, setStepError] = useState<string | null>(null);

  const loadSample = useCallback(
    (filterLabel?: number) => {
      setLoading(true);
      setError(null);
      // /backward-step-through returns stages in reverse-topological order (loss
      // first, input last) and skips data nodes. Reverse them to forward visual
      // order (input left → output right), then prepend a synthetic data node at
      // the very left; start at the output end, where the backward pass begins.
      const body: Record<string, unknown> = { graph: JSON.parse(graphJson) };
      if (filterLabel != null) body.filterLabel = filterLabel;
      fetch(apiUrl('/backward-step-through'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
        .then((r) => r.json())
        .then((data: { status: string; result?: BackwardResult; error?: string }) => {
          if (data.status === 'ok' && data.result) {
            const s = [buildDataStage(data.result.sample), ...[...data.result.stages].reverse()];
            setStages(s);
            setLoss(data.result.loss);
            setSample(data.result.sample);
            setSampleIdx(data.result.sampleIdx);
            setStep(null); // stale for the new sample
            setShowAfter(false);
            setStepError(null);
            setCurrentIdx(Math.max(0, s.length - 1)); // start at the output (backward origin)
            setLoaded(true);
          } else {
            setError(data.error ?? 'Failed to load backprop');
          }
        })
        .catch(() => setError('Cannot connect to backend'))
        .finally(() => setLoading(false));
    },
    [graphJson],
  );

  // Apply one gradient step on the SAME sample (sampleIdx) and show before/after.
  const takeStep = useCallback(() => {
    setStepLoading(true);
    setStepError(null);
    fetch(apiUrl('/one-step'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), sampleIdx }),
    })
      .then((r) => r.json())
      .then((data: { status: string; result?: OneStepResult; error?: string }) => {
        if (data.status === 'ok' && data.result) {
          setStep(data.result);
          setShowAfter(true); // reveal the after-state right away
        } else {
          setStepError(data.error ?? 'Failed to take a step');
        }
      })
      .catch(() => setStepError('Cannot connect to backend'))
      .finally(() => setStepLoading(false));
  }, [graphJson, sampleIdx]);

  // Load once when this page mounts (i.e. becomes the active tab). Empty deps so
  // it fires a single time — on error, `loaded` stays false but the effect won't
  // re-run, so no infinite retry.
  useEffect(() => {
    loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ← / → to step (capture phase so it beats React Flow, like step-through).
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        e.stopPropagation();
        if (e.key === 'ArrowLeft') setCurrentIdx((i) => Math.max(0, i - 1));
        else setCurrentIdx((i) => Math.min(stages.length - 1, i + 1));
      }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [stages.length]);

  const stage = stages[currentIdx];
  // In forward visual order, grad-in arrives from the layer to the RIGHT
  // (downstream, toward the loss); grad-out is passed to the layer on the LEFT.
  const fromName = stages[currentIdx + 1]?.displayName ?? 'the loss';
  const toName = stages[currentIdx - 1]?.displayName ?? null;

  return (
    <div className="backprop-page">
      {error && <div className="step-through-empty">{error}</div>}
      {loading && !loaded && <div className="step-through-empty">Running the graph…</div>}

      {stages.length > 0 && (
        <>
          <div className="step-through-controls">
            <button
              className="step-through-ctrl"
              onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
              disabled={currentIdx === 0}
            >
              &#x25C0;
            </button>
            <input
              type="range"
              min={0}
              max={stages.length - 1}
              value={currentIdx}
              onChange={(e) => setCurrentIdx(parseInt(e.target.value, 10))}
              className="step-through-slider"
            />
            <button
              className="step-through-ctrl"
              onClick={() => setCurrentIdx((i) => Math.min(stages.length - 1, i + 1))}
              disabled={currentIdx >= stages.length - 1}
            >
              &#x25B6;
            </button>
            <span className="step-through-counter">
              {currentIdx + 1} / {stages.length}
            </span>
          </div>

          <StageTimeline
            stages={stages}
            currentIdx={currentIdx}
            onSelect={setCurrentIdx}
            direction="backward"
            align="right"
            dataControls={{
              onNewSample: () => loadSample(),
              onPickLabel: (l) => loadSample(l),
              classNames: sample?.classNames ?? null,
              currentLabel: sample?.actualLabel ?? null,
              loading,
            }}
          />

          {stage && (
            <BackpropDetail
              stage={stage}
              stepLayer={step?.layers.find((l: StepLayer) => l.nodeId === stage.nodeId) ?? null}
              lr={step?.lr ?? null}
              showAfter={showAfter}
              fromName={fromName}
              toName={toName}
              graphJson={graphJson}
              sampleIdx={sampleIdx}
              sample={sample}
            />
          )}

          {/* Step controls + before/after summary, anchored to the bottom so
              toggling never displaces the timeline or the layer cell above. */}
          <div className="backprop-footer">
            <div className="backprop-footer-controls">
              {!step ? (
                <button
                  className="backprop-step-btn"
                  onClick={takeStep}
                  disabled={stepLoading || loading}
                  title="Apply one gradient step and show the before/after effect"
                >
                  {stepLoading ? 'Stepping…' : 'Take one gradient step ▶'}
                </button>
              ) : (
                <div className="bp-ba-toggle" role="group" aria-label="Before / after one step">
                  <button
                    className={!showAfter ? 'bp-ba-active' : ''}
                    onClick={() => setShowAfter(false)}
                  >
                    Before
                  </button>
                  <button
                    className={showAfter ? 'bp-ba-active' : ''}
                    onClick={() => setShowAfter(true)}
                  >
                    After step
                  </button>
                </div>
              )}
            </div>

            <div className="backprop-footer-info">
              {stepError ? (
                <span className="backprop-step-error">{stepError}</span>
              ) : step && showAfter ? (
                <OneStepBanner result={step} />
              ) : (
                <span className="backprop-footer-hint">
                  {step
                    ? 'Showing the weights before the step.'
                    : 'Stepping backward, output → input. ◀ / ▶ or arrow keys to move.'}
                  {loss !== null && (
                    <>
                      {' · loss '}
                      <strong>{loss.toFixed(4)}</strong>
                    </>
                  )}
                </span>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
