// StepThroughPanel — full-screen overlay.
// Architecture: sample is top-level, forward/backward are views of the same sample.

import { useEffect, useState, useRef, useCallback } from 'react';
import type { StepThroughResult, BackwardStepThroughResult, StepThroughMode, DenoiseStepThroughResult, TextGenerationResult, SampleInfo } from './types';
import { StageTimeline } from './StageTimeline';
import { StageDetail } from './StageDetail';
import './StepThroughPanel.css';

interface Props {
  open: boolean;
  graphJson: string;
  onClose: () => void;
}

export function StepThroughPanel({ open, graphJson, onClose }: Props) {
  const [mode, setMode] = useState<StepThroughMode>('forward');

  // Sample state — shared across forward/backward
  const [sampleIdx, setSampleIdx] = useState<number | null>(null);
  const [sample, setSample] = useState<SampleInfo | null>(null);

  // Per-mode results
  const [forwardResult, setForwardResult] = useState<StepThroughResult | null>(null);
  const [backwardResult, setBackwardResult] = useState<BackwardStepThroughResult | null>(null);
  const [denoiseResult, setDenoiseResult] = useState<DenoiseStepThroughResult | null>(null);
  const [genResult, setGenResult] = useState<TextGenerationResult | null>(null);
  const [genPrompt, setGenPrompt] = useState('');
  const [genTemp, setGenTemp] = useState(0.8);
  const [genTopK, setGenTopK] = useState(0);
  const [genMaxTokens, setGenMaxTokens] = useState(200);

  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load a new sample (runs forward pass). Invalidates backward.
  const loadSample = useCallback((filterLabel?: number) => {
    setLoading(true);
    setError(null);
    setBackwardResult(null); // invalidate backward for new sample
    const body: Record<string, unknown> = { graph: JSON.parse(graphJson) };
    if (filterLabel != null) body.filterLabel = filterLabel;
    fetch('http://localhost:8000/step-through', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') {
          setForwardResult(data.result);
          setSample(data.result.sample);
          setSampleIdx(data.result.sampleIdx ?? null);
          setCurrentIdx(0);
          setMode('forward');
        } else {
          setError(data.error ?? 'Failed to load step-through');
        }
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  }, [graphJson]);

  // Load backward for the SAME sample (uses sampleIdx)
  const loadBackward = useCallback(() => {
    setLoading(true);
    setError(null);
    const body: Record<string, unknown> = { graph: JSON.parse(graphJson) };
    if (sampleIdx != null) body.sampleIdx = sampleIdx;
    fetch('http://localhost:8000/backward-step-through', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
  }, [graphJson, sampleIdx]);

  const loadDenoise = useCallback(() => {
    setLoading(true);
    setError(null);
    if (isGANGraph) {
      fetch('http://localhost:8000/gan-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(graphJson), numSamples: 8 }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.status === 'ok') {
            setDenoiseResult({
              steps: [{ timestep: 0, pixels: data.result.images }],
              numTimesteps: 1, numSamples: data.result.numSamples,
              imageH: data.result.imageH, imageW: data.result.imageW, channels: data.result.channels,
            });
            setCurrentIdx(0);
          } else setError(data.error ?? 'Failed to generate images');
        })
        .catch(() => setError('Cannot connect to backend'))
        .finally(() => setLoading(false));
    } else {
      fetch('http://localhost:8000/denoise-step-through', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph: JSON.parse(graphJson), numSamples: 4, captureEvery: Math.max(1, Math.floor(100 / 50)) }),
      })
        .then((r) => r.json())
        .then((data) => {
          if (data.status === 'ok') { setDenoiseResult(data.result); setCurrentIdx(0); }
          else setError(data.error ?? 'Failed to run denoising');
        })
        .catch(() => setError('Cannot connect to backend'))
        .finally(() => setLoading(false));
    }
  }, [graphJson]);

  const loadGenerate = useCallback(() => {
    setLoading(true);
    setError(null);
    fetch('http://localhost:8000/generate-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson), prompt: genPrompt || '', maxTokens: genMaxTokens, temperature: genTemp, topK: genTopK }),
    })
      .then((r) => r.json())
      .then((data) => { if (data.status === 'ok') setGenResult(data.result); else setError(data.error ?? 'Failed to generate text'); })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  }, [graphJson, genPrompt, genMaxTokens, genTemp, genTopK]);

  // Initial load
  useEffect(() => {
    if (open && !forwardResult && !loading) loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Escape to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  // Graph type detection
  const isDiffusionGraph = (() => {
    try { return JSON.parse(graphJson).graph?.nodes?.some((n: any) => n.type === 'ml.diffusion.noise_scheduler') ?? false; }
    catch { return false; }
  })();
  const isGANGraph = (() => {
    try { return JSON.parse(graphJson).graph?.nodes?.some((n: any) => n.type === 'ml.gan.noise_input' || n.type === 'ml.loss.gan') ?? false; }
    catch { return false; }
  })();
  const isAutoregGraph = (() => {
    try { return JSON.parse(graphJson).graph?.nodes?.some((n: any) => n.type === 'data.tiny_shakespeare') ?? false; }
    catch { return false; }
  })();
  const isGenerativeGraph = isDiffusionGraph || isGANGraph;

  // Active stages depend on mode
  const activeStages = mode === 'forward' ? (forwardResult?.stages ?? [])
    : mode === 'backward' ? (backwardResult?.stages ?? [])
    : [];
  const denoiseSteps = denoiseResult?.steps ?? [];
  const totalSteps = mode === 'denoise' ? denoiseSteps.length : activeStages.length;

  // Keyboard nav
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (mode === 'generate') return;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        e.stopPropagation();
        if (e.key === 'ArrowLeft') setCurrentIdx((i) => Math.max(0, i - 1));
        else setCurrentIdx((i) => Math.min(totalSteps - 1, i + 1));
      }
    };
    // Use capture phase so we intercept before React Flow
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [open, mode, totalSteps]);

  const switchMode = (m: StepThroughMode) => {
    setMode(m);
    setCurrentIdx(0);
    // Auto-load backward on first switch (uses same sample)
    if (m === 'backward' && !backwardResult && !loading) loadBackward();
    if (m === 'denoise' && !denoiseResult && !loading) loadDenoise();
  };

  if (!open) return null;
  const stage = activeStages[currentIdx];

  return (
    <div className="step-through-panel">
      {/* Header */}
      <div className="step-through-header">
        <span className="step-through-title">Step-Through</span>
        <div className="step-through-header-actions">
          <button className="step-through-close" onClick={onClose}>&times;</button>
        </div>
      </div>

      {/* Sample section — always visible, top-level */}
      {sample && (
        <SampleHeader
          sample={sample}
          onPickLabel={(label) => loadSample(label)}
          onRandom={() => loadSample()}
          loading={loading}
        />
      )}

      {loading && <div className="step-through-empty">Running forward pass...</div>}
      {error && <div className="step-through-error">{error}</div>}

      {/* Mode tabs — below sample */}
      {(forwardResult || backwardResult) && !loading && (
        <div className="step-through-mode-bar">
          <div className="step-through-mode-tabs">
            <button className={`step-through-mode-tab ${mode === 'forward' ? 'step-through-mode-tab-active' : ''}`} onClick={() => switchMode('forward')}>Forward</button>
            <button className={`step-through-mode-tab ${mode === 'backward' ? 'step-through-mode-tab-active step-through-mode-tab-backward' : ''}`} onClick={() => switchMode('backward')}>Backward</button>
            {isGenerativeGraph && (
              <button className={`step-through-mode-tab ${mode === 'denoise' ? 'step-through-mode-tab-active step-through-mode-tab-denoise' : ''}`} onClick={() => switchMode('denoise')}>{isGANGraph ? 'Generate' : 'Denoise'}</button>
            )}
            {isAutoregGraph && (
              <button className={`step-through-mode-tab ${mode === 'generate' ? 'step-through-mode-tab-active step-through-mode-tab-denoise' : ''}`} onClick={() => switchMode('generate')}>Generate</button>
            )}
          </div>
        </div>
      )}

      {totalSteps > 0 && !loading && (
        <>
          {/* Navigation */}
          {mode !== 'generate' && (
            <div className="step-through-controls">
              <button className="step-through-ctrl" onClick={() => setCurrentIdx((i) => Math.max(0, i - 1))} disabled={currentIdx === 0}>&#x25C0;</button>
              <input type="range" min={0} max={totalSteps - 1} value={currentIdx} onChange={(e) => setCurrentIdx(parseInt(e.target.value, 10))} className="step-through-slider" />
              <button className="step-through-ctrl" onClick={() => setCurrentIdx((i) => Math.min(totalSteps - 1, i + 1))} disabled={currentIdx >= totalSteps - 1}>&#x25B6;</button>
              <span className="step-through-counter">
                {mode === 'denoise' && denoiseSteps[currentIdx] ? `t=${denoiseSteps[currentIdx].timestep}` : `${currentIdx + 1} / ${totalSteps}`}
              </span>
            </div>
          )}

          {/* Forward */}
          {mode === 'forward' && (
            <>
              <StageTimeline stages={activeStages} currentIdx={currentIdx} onSelect={setCurrentIdx} direction="forward" />
              {stage && <StageDetail stage={stage} />}
            </>
          )}

          {/* Backward */}
          {mode === 'backward' && (
            <>
              <StageTimeline stages={activeStages} currentIdx={currentIdx} onSelect={setCurrentIdx} direction="backward" />
              {stage && (
                <div className="stage-detail">
                  <div className="stage-detail-header">
                    <span className="stage-detail-name">{stage.displayName}</span>
                    <span className="stage-detail-shape">
                      {stage.inputShape ? `[${stage.inputShape.join(', ')}]` : '—'} &rarr; {stage.outputShape ? `[${stage.outputShape.join(', ')}]` : '—'}
                    </span>
                  </div>
                  <div className="step-through-backward-placeholder">Backward visualization coming soon</div>
                </div>
              )}
            </>
          )}

          {mode === 'denoise' && denoiseResult && denoiseSteps[currentIdx] && (
            <DenoiseView step={denoiseSteps[currentIdx]} channels={denoiseResult.channels} />
          )}
          {mode === 'generate' && (
            <GenerateView result={genResult} prompt={genPrompt} temperature={genTemp} topK={genTopK} maxTokens={genMaxTokens} loading={loading}
              onPromptChange={setGenPrompt} onTempChange={setGenTemp} onTopKChange={setGenTopK} onMaxTokensChange={setGenMaxTokens} onGenerate={loadGenerate} />
          )}
        </>
      )}

      {/* Backward loading state */}
      {mode === 'backward' && !backwardResult && loading && (
        <div className="step-through-empty">Running backward pass on same sample...</div>
      )}

      {mode === 'generate' && !loading && totalSteps === 0 && (
        <GenerateView result={genResult} prompt={genPrompt} temperature={genTemp} topK={genTopK} maxTokens={genMaxTokens} loading={loading}
          onPromptChange={setGenPrompt} onTempChange={setGenTemp} onTopKChange={setGenTopK} onMaxTokensChange={setGenMaxTokens} onGenerate={loadGenerate} />
      )}
    </div>
  );
}

// --- Sample header (top-level, above mode tabs) ---

function SampleHeader({ sample, onPickLabel, onRandom, loading }: {
  sample: SampleInfo;
  onPickLabel: (label: number) => void;
  onRandom: () => void;
  loading: boolean;
}) {
  const classNames = sample.classNames;
  return (
    <div className="step-through-sample">
      {sample.imagePixels && (
        <SampleImage pixels={sample.imagePixels} channels={sample.imageChannels ?? 1} />
      )}
      <div className="step-through-sample-info">
        {sample.actualLabel != null && (
          <div>
            <span className="step-through-sample-label">Label: </span>
            <span className="step-through-sample-value">
              {classNames ? `${classNames[sample.actualLabel]} (${sample.actualLabel})` : sample.actualLabel}
            </span>
          </div>
        )}
        {sample.datasetType && <div className="step-through-sample-dataset">{sample.datasetType}</div>}
        {sample.sampleText && <div className="step-through-sample-text">{sample.sampleText}</div>}
        {!sample.sampleText && sample.tokenIds && (
          <div className="step-through-sample-tokens">First tokens: {sample.tokenIds.slice(0, 16).join(', ')}...</div>
        )}
      </div>
      <div className="step-through-sample-actions">
        <button className="step-through-btn" onClick={onRandom} disabled={loading}>Random</button>
        {classNames && classNames.length <= 20 && (
          <div className="step-through-label-picker">
            <div className="step-through-label-picker-title">By class</div>
            <div className="step-through-label-picker-btns">
              {classNames.map((name, i) => (
                <button
                  key={i}
                  className={`step-through-label-btn ${sample.actualLabel === i ? 'step-through-label-btn-active' : ''}`}
                  onClick={() => onPickLabel(i)}
                  disabled={loading}
                  title={name}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// --- Generate view ---

function GenerateView({ result, prompt, temperature, topK, maxTokens, loading, onPromptChange, onTempChange, onTopKChange, onMaxTokensChange, onGenerate }: {
  result: TextGenerationResult | null; prompt: string; temperature: number; topK: number; maxTokens: number; loading: boolean;
  onPromptChange: (v: string) => void; onTempChange: (v: number) => void; onTopKChange: (v: number) => void; onMaxTokensChange: (v: number) => void; onGenerate: () => void;
}) {
  return (
    <div className="generate-view">
      <div className="generate-controls">
        <div className="generate-prompt-row">
          <label className="generate-label">Prompt</label>
          <textarea className="generate-prompt-input" value={prompt} onChange={(e) => onPromptChange(e.target.value)} placeholder="Start of generated text (optional)" rows={3} />
        </div>
        <div className="generate-sliders">
          <div className="generate-slider-group">
            <label className="generate-label" title="Controls randomness.">Temperature <span className="generate-label-help">?</span></label>
            <div className="generate-slider-row">
              <input type="range" min={0.1} max={2.0} step={0.05} value={temperature} onChange={(e) => onTempChange(parseFloat(e.target.value))} />
              <input type="number" className="generate-slider-value" min={0.1} max={2.0} step={0.05} value={temperature} onChange={(e) => { const v = parseFloat(e.target.value); if (!isNaN(v)) onTempChange(Math.max(0.1, Math.min(2.0, v))); }} />
            </div>
          </div>
          <div className="generate-slider-group">
            <label className="generate-label" title="Limits to K most likely tokens.">Top-K {topK === 0 && <span style={{ opacity: 0.5 }}>(off)</span>} <span className="generate-label-help">?</span></label>
            <div className="generate-slider-row">
              <input type="range" min={0} max={65} step={1} value={topK} onChange={(e) => onTopKChange(parseInt(e.target.value, 10))} />
              <input type="number" className="generate-slider-value" min={0} max={65} step={1} value={topK} onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) onTopKChange(Math.max(0, Math.min(65, v))); }} />
            </div>
          </div>
          <div className="generate-slider-group">
            <label className="generate-label" title="Max tokens to generate.">Max tokens <span className="generate-label-help">?</span></label>
            <div className="generate-slider-row">
              <input type="range" min={20} max={500} step={10} value={maxTokens} onChange={(e) => onMaxTokensChange(parseInt(e.target.value, 10))} />
              <input type="number" className="generate-slider-value" min={20} max={500} step={10} value={maxTokens} onChange={(e) => { const v = parseInt(e.target.value, 10); if (!isNaN(v)) onMaxTokensChange(Math.max(20, Math.min(500, v))); }} />
            </div>
          </div>
        </div>
        <button className="step-through-btn generate-btn" onClick={onGenerate} disabled={loading}>{loading ? 'Generating...' : 'Generate'}</button>
      </div>
      {result && (
        <div className="generate-output">
          <pre className="generate-text"><span className="generate-text-prompt">{result.prompt}</span><span className="generate-text-continuation">{result.generated}</span></pre>
          <div className="generate-meta">{result.tokens.length} tokens generated</div>
        </div>
      )}
    </div>
  );
}

// --- Denoise view ---

function DenoiseView({ step, channels }: { step: { timestep: number; pixels: (number[][] | number[][][])[] }; channels: number }) {
  return (
    <div className="denoise-view">
      <div className="denoise-info">
        <span className="denoise-timestep">Timestep {step.timestep}</span>
        <span className="denoise-hint">{step.timestep > 0 ? 'Denoising in progress' : 'Fully denoised'}</span>
      </div>
      <div className="denoise-samples">{step.pixels.map((pixels, i) => <DenoiseImage key={i} pixels={pixels} channels={channels} />)}</div>
    </div>
  );
}

function DenoiseImage({ pixels, channels }: { pixels: number[][] | number[][][]; channels: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length; const firstRow = pixels[0]; const isRGB = Array.isArray(firstRow[0]);
    const w = isRGB ? (firstRow as number[][]).length : (firstRow as number[]).length;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const data = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      if (isRGB) { const px = (pixels as number[][][])[y][x]; data.data[idx] = px[0]; data.data[idx+1] = px[1]; data.data[idx+2] = px[2]; }
      else { const v = (pixels as number[][])[y][x]; data.data[idx] = v; data.data[idx+1] = v; data.data[idx+2] = v; }
      data.data[idx+3] = 255;
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, channels]);
  return <canvas ref={canvasRef} className="denoise-sample-img" />;
}

function SampleImage({ pixels, channels }: { pixels: number[][] | number[][][]; channels: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length; const firstRow = pixels[0] as (number[] | number[][]);
    const w = Array.isArray((firstRow as number[][])[0]) ? (firstRow as number[][]).length : (firstRow as number[]).length;
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const data = ctx.createImageData(w, h); const isRGB = channels >= 3;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      if (isRGB) { const px = (pixels as number[][][])[y][x]; data.data[idx] = px[0]; data.data[idx+1] = px[1]; data.data[idx+2] = px[2]; }
      else { const v = (pixels as number[][])[y][x]; data.data[idx] = v; data.data[idx+1] = v; data.data[idx+2] = v; }
      data.data[idx+3] = 255;
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, channels]);
  return <canvas ref={canvasRef} className="step-through-sample-img" />;
}
