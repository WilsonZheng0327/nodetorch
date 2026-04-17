// StepThroughPanel — bottom drawer that shows the forward pass as a timeline.
// Loads a sample from the dataset and walks through each layer's transformation.

import { useEffect, useState, useRef } from 'react';
import type { StepThroughResult } from './types';
import { StageTimeline } from './StageTimeline';
import { StageDetail } from './StageDetail';
import { PerturbCanvas } from './PerturbCanvas';
import './StepThroughPanel.css';

interface Props {
  open: boolean;
  graphJson: string;
  onClose: () => void;
}

export function StepThroughPanel({ open, graphJson, onClose }: Props) {
  const [result, setResult] = useState<StepThroughResult | null>(null);
  const [resultB, setResultB] = useState<StepThroughResult | null>(null);  // second sample for compare
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [perturbMode, setPerturbMode] = useState(false);
  const [mask, setMask] = useState<number[][] | null>(null);

  // Load a new sample (either replacing A, or loading B for compare)
  // If `mask` is provided, sent as perturbation; fresh sample otherwise.
  const loadSample = (asCompare = false, withMask: number[][] | null = null) => {
    setLoading(true);
    setError(null);
    const body: any = { graph: JSON.parse(graphJson) };
    if (withMask) body.mask = withMask;
    fetch('http://localhost:8000/step-through', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') {
          if (asCompare) {
            setResultB(data.result);
          } else {
            setResult(data.result);
            setResultB(null);
            setCurrentIdx(0);
            if (!withMask) {
              setMask(null);
              setPerturbMode(false);
            }
          }
        } else {
          setError(data.error ?? 'Failed to load step-through');
        }
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  };

  const exitCompare = () => setResultB(null);
  const compareMode = resultB !== null;

  // Initial load when opened
  useEffect(() => {
    if (open && !result && !loading) loadSample(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Auto-play logic
  const playTimerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!isPlaying || !result) return;
    playTimerRef.current = window.setInterval(() => {
      setCurrentIdx((i) => {
        if (i >= result.stages.length - 1) {
          setIsPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 800);
    return () => {
      if (playTimerRef.current) window.clearInterval(playTimerRef.current);
    };
  }, [isPlaying, result]);

  if (!open) return null;

  const stage = result?.stages[currentIdx];

  return (
    <div className="step-through-panel">
      <div className="step-through-header">
        <span className="step-through-title">
          Step-Through
          {result?.modelState && (
            <span
              className={`step-through-model-state ${result.modelState.usingTrainedWeights ? 'step-through-model-state-trained' : ''}`}
              title={result.modelState.note}
            >
              {result.modelState.usingTrainedWeights ? 'Trained' : 'Random weights'}
            </span>
          )}
        </span>
        <div className="step-through-header-actions">
          <button className="step-through-btn" onClick={() => loadSample(false)} disabled={loading}>
            Load Random Sample
          </button>
          {result && !compareMode && (
            <button className="step-through-btn" onClick={() => loadSample(true)} disabled={loading}>
              + Compare
            </button>
          )}
          {compareMode && (
            <button className="step-through-btn" onClick={exitCompare}>
              Exit Compare
            </button>
          )}
          <button className="step-through-close" onClick={onClose}>&times;</button>
        </div>
      </div>

      {loading && (
        <div className="step-through-empty">Loading sample and running forward pass...</div>
      )}
      {error && <div className="step-through-error">{error}</div>}

      {result && !loading && (
        <>
          {compareMode && resultB ? (
            <div className="step-through-compare-samples">
              <SampleHeader sample={result.sample} label="A" />
              <SampleHeader sample={resultB.sample} label="B" />
            </div>
          ) : perturbMode && result.sample.imagePixels ? (
            <PerturbHeader
              sample={result.sample}
              mask={mask}
              onMaskChange={setMask}
              onApply={() => mask && loadSample(false, mask)}
              onClear={() => setMask(null)}
              onExit={() => { setPerturbMode(false); setMask(null); loadSample(false); }}
            />
          ) : (
            <SampleHeader
              sample={result.sample}
              onStartPerturb={result.sample.imagePixels ? () => setPerturbMode(true) : undefined}
            />
          )}

          <div className="step-through-controls">
            <button
              className="step-through-ctrl"
              onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
              disabled={currentIdx === 0}
            >
              ◀
            </button>
            <button
              className={`step-through-ctrl ${isPlaying ? 'step-through-ctrl-playing' : ''}`}
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>
            <button
              className="step-through-ctrl"
              onClick={() => setCurrentIdx((i) => Math.min(result.stages.length - 1, i + 1))}
              disabled={currentIdx >= result.stages.length - 1}
            >
              ▶
            </button>
            <input
              type="range"
              min={0}
              max={result.stages.length - 1}
              value={currentIdx}
              onChange={(e) => setCurrentIdx(parseInt(e.target.value, 10))}
              className="step-through-slider"
            />
            <span className="step-through-counter">
              {currentIdx + 1} / {result.stages.length}
            </span>
          </div>

          <StageTimeline
            stages={result.stages}
            currentIdx={currentIdx}
            onSelect={setCurrentIdx}
          />

          {stage && (
            compareMode && resultB ? (
              <div className="step-through-compare-details">
                <div className="step-through-compare-pane">
                  <div className="step-through-compare-label">A</div>
                  <StageDetail stage={stage} />
                </div>
                <div className="step-through-compare-pane">
                  <div className="step-through-compare-label">B</div>
                  {resultB.stages[currentIdx] && <StageDetail stage={resultB.stages[currentIdx]} />}
                </div>
              </div>
            ) : (
              <StageDetail stage={stage} />
            )
          )}
        </>
      )}
    </div>
  );
}

// --- Sample preview header ---

function SampleHeader({ sample, label, onStartPerturb }: {
  sample: StepThroughResult['sample'];
  label?: string;
  onStartPerturb?: () => void;
}) {
  return (
    <div className="step-through-sample">
      {label && <div className="step-through-sample-tag">{label}</div>}
      {sample.imagePixels && (
        <SampleImage pixels={sample.imagePixels} channels={sample.imageChannels ?? 1} />
      )}
      <div className="step-through-sample-info">
        {sample.actualLabel != null && (
          <div>
            <span className="step-through-sample-label">Actual label: </span>
            <span className="step-through-sample-value">{sample.actualLabel}</span>
          </div>
        )}
        {sample.datasetType && (
          <div className="step-through-sample-dataset">{sample.datasetType}</div>
        )}
        {sample.tokenIds && (
          <div className="step-through-sample-tokens">
            First tokens: {sample.tokenIds.slice(0, 16).join(', ')}…
          </div>
        )}
      </div>
      {onStartPerturb && (
        <button className="step-through-btn" onClick={onStartPerturb} title="Draw a mask on the input to see how the model responds">
          Perturb
        </button>
      )}
    </div>
  );
}

// --- Perturb mode header (drawing UI) ---

function PerturbHeader({
  sample,
  mask,
  onMaskChange,
  onApply,
  onClear,
  onExit,
}: {
  sample: StepThroughResult['sample'];
  mask: number[][] | null;
  onMaskChange: (mask: number[][]) => void;
  onApply: () => void;
  onClear: () => void;
  onExit: () => void;
}) {
  if (!sample.imagePixels) return null;
  return (
    <div className="step-through-sample step-through-sample-perturb">
      <PerturbCanvas
        pixels={sample.imagePixels}
        channels={sample.imageChannels ?? 1}
        mask={mask}
        onMaskChange={onMaskChange}
        displaySize={96}
      />
      <div className="step-through-sample-info">
        <div className="step-through-perturb-hint">
          Draw to mask regions. Masked pixels will be zeroed before re-running the model.
        </div>
        <div className="step-through-perturb-actions">
          <button className="step-through-btn" onClick={onApply} disabled={!mask}>
            Apply &amp; Re-run
          </button>
          <button className="step-through-btn" onClick={onClear}>
            Clear
          </button>
          <button className="step-through-btn" onClick={onExit}>
            Exit Perturb
          </button>
        </div>
      </div>
    </div>
  );
}

function SampleImage({ pixels, channels }: { pixels: number[][] | number[][][]; channels: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length;
    const firstRow = pixels[0] as (number[] | number[][]);
    const w = Array.isArray((firstRow as number[][])[0]) ? (firstRow as number[][]).length : (firstRow as number[]).length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
    const isRGB = channels >= 3;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (isRGB) {
          const px = (pixels as number[][][])[y][x];
          data.data[idx] = px[0];
          data.data[idx + 1] = px[1];
          data.data[idx + 2] = px[2];
        } else {
          const v = (pixels as number[][])[y][x];
          data.data[idx] = v;
          data.data[idx + 1] = v;
          data.data[idx + 2] = v;
        }
        data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, channels]);
  return <canvas ref={canvasRef} className="step-through-sample-img" />;
}
