// StepThroughPanel — bottom drawer that shows the forward pass as a timeline.
// Loads a sample from the dataset and walks through each layer's transformation.

import { useEffect, useState, useRef } from 'react';
import type { StepThroughResult } from './types';
import { StageTimeline } from './StageTimeline';
import { StageDetail } from './StageDetail';
import './StepThroughPanel.css';

interface Props {
  open: boolean;
  graphJson: string;
  onClose: () => void;
}

export function StepThroughPanel({ open, graphJson, onClose }: Props) {
  const [result, setResult] = useState<StepThroughResult | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Load a new sample
  const loadSample = () => {
    setLoading(true);
    setError(null);
    fetch('http://localhost:8000/step-through', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ graph: JSON.parse(graphJson) }),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.status === 'ok') {
          setResult(data.result);
          setCurrentIdx(0);
        } else {
          setError(data.error ?? 'Failed to load step-through');
        }
      })
      .catch(() => setError('Cannot connect to backend'))
      .finally(() => setLoading(false));
  };

  // Initial load when opened
  useEffect(() => {
    if (open && !result && !loading) loadSample();
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
        <span className="step-through-title">Step-Through</span>
        <div className="step-through-header-actions">
          <button className="step-through-btn" onClick={loadSample} disabled={loading}>
            Load Random Sample
          </button>
          <button className="step-through-close" onClick={onClose}>&times;</button>
        </div>
      </div>

      {loading && (
        <div className="step-through-empty">Loading sample and running forward pass...</div>
      )}
      {error && <div className="step-through-error">{error}</div>}

      {result && !loading && (
        <>
          <SampleHeader sample={result.sample} />

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

          {stage && <StageDetail stage={stage} />}
        </>
      )}
    </div>
  );
}

// --- Sample preview header ---

function SampleHeader({ sample }: { sample: StepThroughResult['sample'] }) {
  return (
    <div className="step-through-sample">
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
