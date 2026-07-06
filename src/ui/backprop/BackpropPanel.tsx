// BackpropPanel — fullscreen overlay for stepping through the backward pass.
// The layer timeline is right-aligned and reads right → left (output → input),
// mirroring the forward step-through frame. Each stage shows the real backward
// data for that layer: the gradient arriving (dL/d-output) and the weight-gradient
// it produces. Before/after-one-step and wiggle-a-weight land in later phases.

import { useEffect, useState, useCallback } from 'react';
import type { BackwardResult, BackwardStage, OneStepResult, StepLayer } from './types';
import { StageTimeline } from '../step-through/StageTimeline';
import { BackpropDetail } from './BackpropDetail';
import { OneStepBanner } from './OneStepBanner';
import '../step-through/StepThroughPanel.css';
import './BackpropPanel.css';
import { apiUrl } from '../../api/base';

interface Props {
  open: boolean;
  graphJson: string;
  onClose: () => void;
}

export function BackpropPanel({ open, graphJson, onClose }: Props) {
  const [stages, setStages] = useState<BackwardStage[]>([]);
  const [loss, setLoss] = useState<number | null>(null);
  const [sampleIdx, setSampleIdx] = useState<number | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  // One-step (before/after) state — computed lazily on demand.
  const [step, setStep] = useState<OneStepResult | null>(null);
  const [stepLoading, setStepLoading] = useState(false);
  const [stepError, setStepError] = useState<string | null>(null);

  const loadSample = useCallback(() => {
    setLoading(true);
    setError(null);
    // /backward-step-through returns stages in reverse-topological order (loss
    // first, input last) and skips data nodes. Reverse them back to forward
    // visual order (input left → output right) so the right-aligned timeline
    // reads the same as the forward step-through; then start at the output end,
    // where the backward pass originates.
    fetch(apiUrl('/backward-step-through'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson) }),
    })
      .then((r) => r.json())
      .then((data: { status: string; result?: BackwardResult; error?: string }) => {
        if (data.status === 'ok' && data.result) {
          const s = [...data.result.stages].reverse();
          setStages(s);
          setLoss(data.result.loss);
          setSampleIdx(data.result.sampleIdx);
          setStep(null); // stale for the new sample
          setStepError(null);
          setCurrentIdx(Math.max(0, s.length - 1)); // start at the output (backward origin)
          setLoaded(true);
        } else {
          setError(data.error ?? 'Failed to load backprop');
        }
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  }, [graphJson]);

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
        if (data.status === 'ok' && data.result) setStep(data.result);
        else setStepError(data.error ?? 'Failed to take a step');
      })
      .catch(() => setStepError('Cannot connect to backend'))
      .finally(() => setStepLoading(false));
  }, [graphJson, sampleIdx]);

  // Load once when the panel opens. Deps are intentionally just [open]: on
  // error `loaded` stays false and `loading` flips back to false, so including
  // them would re-fire this into an infinite retry loop.
  useEffect(() => {
    if (open && !loaded && !loading) loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // ← / → to step (capture phase so it beats React Flow, like step-through).
  useEffect(() => {
    if (!open) return;
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
  }, [open, stages.length]);

  if (!open) return null;
  const stage = stages[currentIdx];

  return (
    <div className="step-through-panel">
      <div className="step-through-header">
        <span className="step-through-title">Backprop</span>
        <div className="step-through-header-actions">
          <button className="step-through-close" onClick={onClose}>
            &times;
          </button>
        </div>
      </div>

      {loaded && (
        <div className="backprop-sample-bar">
          <button className="backprop-sample-btn" onClick={loadSample} disabled={loading}>
            {loading ? 'Loading…' : 'New sample'}
          </button>
          <span className="backprop-sample-hint">
            Stepping backward, output → input. ◀ / ▶ or arrow keys to move.
          </span>
          {loss !== null && (
            <span className="backprop-loss">
              loss <strong>{loss.toFixed(4)}</strong>
            </span>
          )}
          <button
            className="backprop-step-btn"
            onClick={takeStep}
            disabled={stepLoading || loading}
            title="Apply one gradient step and show the before/after effect"
          >
            {stepLoading ? 'Stepping…' : step ? 'Re-run step' : 'Take one gradient step ▶'}
          </button>
        </div>
      )}

      {stepError && <div className="backprop-step-error">{stepError}</div>}
      {step && <OneStepBanner result={step} />}

      {error && <div className="step-through-empty">{error}</div>}
      {loading && !loaded && <div className="step-through-empty">Running the graph…</div>}

      {stages.length > 0 && (
        <>
          <StageTimeline
            stages={stages}
            currentIdx={currentIdx}
            onSelect={setCurrentIdx}
            direction="backward"
            align="right"
          />

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

          {stage && (
            <BackpropDetail
              stage={stage}
              stepLayer={step?.layers.find((l: StepLayer) => l.nodeId === stage.nodeId) ?? null}
              lr={step?.lr ?? null}
            />
          )}
        </>
      )}
    </div>
  );
}
