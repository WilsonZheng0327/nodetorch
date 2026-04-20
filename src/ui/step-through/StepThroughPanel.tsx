// StepThroughPanel — bottom drawer that shows the forward pass as a timeline.
// Loads a sample from the dataset and walks through each layer's transformation.

import { useEffect, useState, useRef, useCallback } from 'react';
import type { StepThroughResult, BackwardStepThroughResult, StepThroughMode, DenoiseStepThroughResult } from './types';
import { StageTimeline } from './StageTimeline';
import { StageCard } from './StageCard';
import { StageDetail } from './StageDetail';
import { PerturbCanvas } from './PerturbCanvas';
import './StepThroughPanel.css';

interface Props {
  open: boolean;
  graphJson: string;
  onClose: () => void;
}

export function StepThroughPanel({ open, graphJson, onClose }: Props) {
  const [mode, setMode] = useState<StepThroughMode>('forward');
  const [result, setResult] = useState<StepThroughResult | null>(null);
  const [resultB, setResultB] = useState<StepThroughResult | null>(null);  // second sample for compare
  const [backwardResult, setBackwardResult] = useState<BackwardStepThroughResult | null>(null);
  const [denoiseResult, setDenoiseResult] = useState<DenoiseStepThroughResult | null>(null);
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

  // Load backward step-through
  const loadBackward = () => {
    setLoading(true);
    setError(null);
    fetch('http://localhost:8000/backward-step-through', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson) }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') {
          setBackwardResult(data.result);
          setCurrentIdx(0);
        } else {
          setError(data.error ?? 'Failed to run backward step-through');
        }
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  };

  // Load denoise step-through (diffusion models)
  const loadDenoise = () => {
    setLoading(true);
    setError(null);

    if (isGANGraph) {
      // GAN: generate images directly (no timestep stepping)
      fetch('http://localhost:8000/gan-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(graphJson), numSamples: 8 }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.status === 'ok') {
            // Wrap in a single-step denoise result so the UI can display it
            setDenoiseResult({
              steps: [{ timestep: 0, pixels: data.result.images }],
              numTimesteps: 1,
              numSamples: data.result.numSamples,
              imageH: data.result.imageH,
              imageW: data.result.imageW,
              channels: data.result.channels,
            });
            setCurrentIdx(0);
          } else {
            setError(data.error ?? 'Failed to generate images');
          }
        })
        .catch(() => setError('Cannot connect to backend'))
        .finally(() => setLoading(false));
    } else {
      // Diffusion: run full denoising step-through
      const captureEvery = Math.max(1, Math.floor(100 / 50));
      fetch('http://localhost:8000/denoise-step-through', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(graphJson), numSamples: 4, captureEvery }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.status === 'ok') {
            setDenoiseResult(data.result);
            setCurrentIdx(0);
          } else {
            setError(data.error ?? 'Failed to run denoising');
          }
        })
        .catch(() => setError('Cannot connect to backend'))
        .finally(() => setLoading(false));
    }
  };

  const exitCompare = () => setResultB(null);
  const compareMode = resultB !== null;

  // Initial load when opened
  useEffect(() => {
    if (open && !result && !loading) loadSample(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Escape to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  // Detect if this is a diffusion graph
  const isDiffusionGraph = (() => {
    try {
      const g = JSON.parse(graphJson);
      return g.graph?.nodes?.some((n: any) => n.type === 'ml.diffusion.noise_scheduler') ?? false;
    } catch { return false; }
  })();

  const isGANGraph = (() => {
    try {
      const g = JSON.parse(graphJson);
      return g.graph?.nodes?.some((n: any) => n.type === 'ml.gan.noise_input' || n.type === 'ml.loss.gan') ?? false;
    } catch { return false; }
  })();

  const isGenerativeGraph = isDiffusionGraph || isGANGraph;

  // The active result and stages depend on the mode
  const activeResult = mode === 'forward' ? result : backwardResult;
  const activeStages = activeResult?.stages ?? [];
  const denoiseSteps = denoiseResult?.steps ?? [];
  const totalSteps = mode === 'denoise' ? denoiseSteps.length : activeStages.length;

  // Auto-play logic
  const playTimerRef = useRef<number | null>(null);
  useEffect(() => {
    if (!isPlaying || totalSteps === 0) return;
    playTimerRef.current = window.setInterval(() => {
      setCurrentIdx((i) => {
        if (i >= totalSteps - 1) {
          setIsPlaying(false);
          return i;
        }
        return i + 1;
      });
    }, 800);
    return () => {
      if (playTimerRef.current) window.clearInterval(playTimerRef.current);
    };
  }, [isPlaying, totalSteps]);

  // When switching modes, auto-load if needed
  const switchMode = (m: StepThroughMode) => {
    setMode(m);
    setCurrentIdx(0);
    setIsPlaying(false);
    if (m === 'backward' && !backwardResult && !loading) loadBackward();
    if (m === 'denoise' && !denoiseResult && !loading) loadDenoise();
  };

  if (!open) return null;

  const stage = activeStages[currentIdx];
  const modelState = activeResult?.modelState ?? result?.modelState;

  return (
    <div className="step-through-panel">
      <div className="step-through-header">
        <span className="step-through-title">
          Step-Through
          <span className="step-through-mode-tabs">
            <button
              className={`step-through-mode-tab ${mode === 'forward' ? 'step-through-mode-tab-active' : ''}`}
              onClick={() => switchMode('forward')}
            >
              Forward
            </button>
            <button
              className={`step-through-mode-tab ${mode === 'backward' ? 'step-through-mode-tab-active step-through-mode-tab-backward' : ''}`}
              onClick={() => switchMode('backward')}
            >
              Backward
            </button>
            {isGenerativeGraph && (
              <button
                className={`step-through-mode-tab ${mode === 'denoise' ? 'step-through-mode-tab-active step-through-mode-tab-denoise' : ''}`}
                onClick={() => switchMode('denoise')}
              >
                {isGANGraph ? 'Generate' : 'Denoise'}
              </button>
            )}
          </span>
          {modelState && (
            <span
              className={`step-through-model-state ${modelState.usingTrainedWeights ? 'step-through-model-state-trained' : ''}`}
              title={modelState.note}
            >
              {modelState.usingTrainedWeights ? 'Trained' : 'Random weights'}
            </span>
          )}
        </span>
        <div className="step-through-header-actions">
          {mode === 'forward' && (
            <>
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
            </>
          )}
          {mode === 'backward' && (
            <button className="step-through-btn" onClick={loadBackward} disabled={loading}>
              Re-run Backward
            </button>
          )}
          {mode === 'denoise' && (
            <button className="step-through-btn" onClick={loadDenoise} disabled={loading}>
              Regenerate
            </button>
          )}
          <button className="step-through-close" onClick={onClose}>&times;</button>
        </div>
      </div>

      {loading && (
        <div className="step-through-empty">
          {mode === 'forward' ? 'Loading sample and running forward pass...' : 'Running forward + backward pass...'}
        </div>
      )}
      {error && <div className="step-through-error">{error}</div>}

      {totalSteps > 0 && !loading && (
        <>
          {/* Sample header — forward mode only */}
          {mode === 'forward' && result && (
            compareMode && resultB ? (
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
                onClear={() => {
                  const h = result.sample.imagePixels!.length;
                  const row0 = result.sample.imagePixels![0];
                  const w = Array.isArray(row0[0]) ? (row0 as number[][]).length : (row0 as number[]).length;
                  const empty: number[][] = Array.from({ length: h }, () => new Array(w).fill(0));
                  setMask(empty);
                }}
                onExit={() => { setPerturbMode(false); setMask(null); }}
              />
            ) : (
              <SampleHeader
                sample={result.sample}
                onStartPerturb={result.sample.imagePixels ? () => setPerturbMode(true) : undefined}
              />
            )
          )}

          {/* Backward mode header — show loss and sample preview */}
          {mode === 'backward' && backwardResult && (
            <div className="step-through-sample">
              {backwardResult.sample?.imagePixels && (
                <SampleImage pixels={backwardResult.sample.imagePixels} channels={backwardResult.sample.imageChannels ?? 1} />
              )}
              <div className="step-through-sample-info">
                <div>
                  <span className="step-through-sample-label">Loss: </span>
                  <span className="step-through-sample-value">{backwardResult.loss.toFixed(4)}</span>
                </div>
                {backwardResult.sample?.actualLabel != null && (
                  <div>
                    <span className="step-through-sample-label">Label: </span>
                    <span className="step-through-sample-value">{backwardResult.sample.actualLabel}</span>
                  </div>
                )}
                {backwardResult.sample?.sampleText && (
                  <div className="step-through-sample-text">{backwardResult.sample.sampleText}</div>
                )}
              </div>
            </div>
          )}

          <div className="step-through-controls">
            <button
              className="step-through-ctrl"
              onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))}
              disabled={currentIdx === 0}
              title="Previous step"
            >
              ⏮
            </button>
            <button
              className={`step-through-ctrl ${isPlaying ? 'step-through-ctrl-playing' : ''}`}
              onClick={() => setIsPlaying(!isPlaying)}
              title={isPlaying ? 'Pause' : 'Play'}
            >
              {isPlaying ? '⏸' : '▶'}
            </button>
            <button
              className="step-through-ctrl"
              onClick={() => setCurrentIdx((i) => Math.min(totalSteps - 1, i + 1))}
              disabled={currentIdx >= totalSteps - 1}
              title="Next step"
            >
              ⏭
            </button>
            <input
              type="range"
              min={0}
              max={totalSteps - 1}
              value={currentIdx}
              onChange={(e) => setCurrentIdx(parseInt(e.target.value, 10))}
              className="step-through-slider"
            />
            <span className="step-through-counter">
              {mode === 'denoise' && denoiseSteps[currentIdx]
                ? `t=${denoiseSteps[currentIdx].timestep}`
                : `${currentIdx + 1} / ${totalSteps}`}
            </span>
          </div>

          {mode !== 'denoise' && (
            <>
              {mode === 'forward' && compareMode && resultB ? (
                <CompareTimeline
                  stagesA={activeStages}
                  stagesB={resultB.stages}
                  currentIdx={currentIdx}
                  onSelect={setCurrentIdx}
                />
              ) : (
                <StageTimeline
                  stages={activeStages}
                  currentIdx={currentIdx}
                  onSelect={setCurrentIdx}
                  direction={mode}
                />
              )}

              {stage && (
                mode === 'forward' && compareMode && resultB ? (
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

          {mode === 'denoise' && denoiseResult && denoiseSteps[currentIdx] && (
            <DenoiseView step={denoiseSteps[currentIdx]} channels={denoiseResult.channels} />
          )}
        </>
      )}
    </div>
  );
}

// --- Sample preview header ---

// --- Compare timeline (two rows, one scrollbar) ---

function CompareTimeline({ stagesA, stagesB, currentIdx, onSelect }: {
  stagesA: any[];
  stagesB: any[];
  currentIdx: number;
  onSelect: (idx: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const active = container.querySelector('.stage-card-active') as HTMLElement | null;
    if (active) {
      active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [currentIdx]);

  const maxLen = Math.max(stagesA.length, stagesB.length);

  return (
    <div className="step-through-compare-timelines" ref={containerRef}>
      <div className="step-through-compare-timelines-inner">
        {Array.from({ length: maxLen }, (_, i) => (
          <div key={i} className="step-through-compare-col">
            <div className="step-through-compare-col-cell">
              {stagesA[i] && (
                <div className="step-through-compare-card-wrap">
                  {i === 0 && <span className="step-through-compare-row-label">A</span>}
                  <StageCard stage={stagesA[i]} active={i === currentIdx} onClick={() => onSelect(i)} />
                </div>
              )}
            </div>
            <div className="step-through-compare-col-cell">
              {stagesB[i] && (
                <div className="step-through-compare-card-wrap">
                  {i === 0 && <span className="step-through-compare-row-label">B</span>}
                  <StageCard stage={stagesB[i]} active={i === currentIdx} onClick={() => onSelect(i)} />
                </div>
              )}
            </div>
            {i < maxLen - 1 && <div className="step-through-compare-arrow">→</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

// --- Denoise view (diffusion step-through) ---

function DenoiseView({ step, channels }: { step: { timestep: number; pixels: (number[][] | number[][][])[] }; channels: number }) {
  return (
    <div className="denoise-view">
      <div className="denoise-info">
        <span className="denoise-timestep">Timestep {step.timestep}</span>
        <span className="denoise-hint">
          {step.timestep > 0 ? 'Denoising in progress — noise is being predicted and removed' : 'Fully denoised — final generated image'}
        </span>
      </div>
      <div className="denoise-samples">
        {step.pixels.map((pixels, i) => (
          <DenoiseImage key={i} pixels={pixels} channels={channels} />
        ))}
      </div>
    </div>
  );
}

function DenoiseImage({ pixels, channels }: { pixels: number[][] | number[][][]; channels: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length;
    const firstRow = pixels[0];
    const isRGB = Array.isArray(firstRow[0]);
    const w = isRGB ? (firstRow as number[][]).length : (firstRow as number[]).length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        if (isRGB) {
          const px = (pixels as number[][][])[y][x];
          data.data[idx] = px[0]; data.data[idx + 1] = px[1]; data.data[idx + 2] = px[2];
        } else {
          const v = (pixels as number[][])[y][x];
          data.data[idx] = v; data.data[idx + 1] = v; data.data[idx + 2] = v;
        }
        data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, channels]);
  return <canvas ref={canvasRef} className="denoise-sample-img" />;
}


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
        {sample.sampleText && (
          <div className="step-through-sample-text">{sample.sampleText}</div>
        )}
        {!sample.sampleText && sample.tokenIds && (
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
