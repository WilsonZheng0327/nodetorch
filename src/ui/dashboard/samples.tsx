// Samples-tab views: tracked classification/reconstruction samples, generated
// text (autoregressive), and generated images (GAN / diffusion). Styles come
// from TrainingDashboard.css.

import { useRef, useEffect } from 'react';
import type { EpochData, TrackedSampleProbe } from './types';

// --- Generated text (autoregressive) ---

export function GeneratedTextView({ progress }: { progress: EpochData[] }) {
  const samples = progress
    .filter((ep) => ep.generatedText)
    .map((ep) => ({ epoch: ep.epoch, text: ep.generatedText! }));

  if (samples.length === 0) {
    return <div className="dashboard-chart-placeholder">No generated samples yet — text is generated periodically during training</div>;
  }

  return (
    <div className="generated-text-samples">
      {samples.map((s) => (
        <div key={s.epoch} className="generated-text-sample">
          <div className="generated-text-epoch">Epoch {s.epoch}</div>
          <pre className="generated-text-content">{s.text}</pre>
        </div>
      ))}
    </div>
  );
}

// --- Tracked samples (classification / reconstruction) ---

export function TrackedSamplesView({ progress, selectedEpoch }: { progress: EpochData[]; selectedEpoch: number | null }) {
  // For GAN/diffusion, show generated samples instead of tracked samples
  const isGenerative = progress.some(ep => ep.generatedSamples?.length);
  if (isGenerative) {
    return <GeneratedSamplesView progress={progress} />;
  }

  if (progress.length === 0 || !progress[0].trackedSamples?.length) {
    return <div className="dashboard-chart-placeholder">No tracked samples — train to see results</div>;
  }

  // Get sample info from the first epoch (images don't change)
  const firstProbes = progress[0].trackedSamples!;

  return (
    <div className="tracked-samples-view">
      {firstProbes.map((sample, sIdx) => (
        <TrackedSampleRow
          key={sample.idx}
          sampleIdx={sIdx}
          sample={sample}
          probes={progress.map(ep => ep.trackedSamples?.[sIdx])}
          epochs={progress.map(ep => ep.epoch)}
          selectedEpoch={selectedEpoch}
        />
      ))}
    </div>
  );
}

function TrackedSampleRow({ sample, probes, epochs, selectedEpoch }: {
  sampleIdx: number;
  sample: TrackedSampleProbe;
  probes: (TrackedSampleProbe | undefined)[];
  epochs: number[];
  selectedEpoch: number | null;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Draw the sample image thumbnail
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sample.imagePixels) return;
    const pixels = sample.imagePixels;
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
  }, [sample.imagePixels]);

  // The epoch in view: the slider selection (1-based) or the latest.
  const selIdx = selectedEpoch != null
    ? Math.min(Math.max(selectedEpoch - 1, 0), probes.length - 1)
    : probes.length - 1;
  const probe = probes[selIdx];
  const epochNum = epochs[selIdx];
  // Classification vs reconstruction is a fixed property of the model.
  const isClassification = (probes[probes.length - 1] ?? probe)?.probabilities != null;

  const correct = probe?.predictedClass === sample.label;
  const conf = (probe?.confidence ?? 0) * 100;
  const maxLoss = Math.max(...probes.filter(Boolean).map(p => p!.loss ?? 0), 0.01);
  const lossPct = ((probe?.loss ?? 0) / maxLoss) * 100;

  return (
    <div className="tracked-sample-row">
      {/* Thumbnail */}
      <div className="tracked-sample-thumb">
        {sample.imagePixels ? (
          <canvas ref={canvasRef} className="tracked-sample-img" />
        ) : (
          <div className="tracked-sample-no-img">#{sample.idx}</div>
        )}
        <div className="tracked-sample-label">Label: {sample.label ?? '?'}</div>
      </div>

      {/* Selected-epoch prediction — one horizontal bar that scrubs with the slider */}
      <div className="tracked-sample-timeline">
        {isClassification ? (
          <>
            <div className="tracked-sample-timeline-header">
              <span>Confidence (epoch {epochNum})</span>
              <span className="tracked-sample-latest">
                {probe?.predictedClass != null
                  ? <>Predicted: {probe.predictedClass} ({conf.toFixed(1)}%)</>
                  : 'no data'}
              </span>
            </div>
            <div className="tracked-sample-hbar-track" title={`Epoch ${epochNum}: class ${probe?.predictedClass ?? '?'} (${conf.toFixed(1)}%)`}>
              <div
                className={`tracked-sample-hbar-fill ${correct ? 'tracked-sample-bar-correct' : 'tracked-sample-bar-wrong'}`}
                style={{ width: `${conf}%` }}
              />
            </div>
          </>
        ) : (
          <>
            <div className="tracked-sample-timeline-header">
              <span>Loss (epoch {epochNum})</span>
              <span className="tracked-sample-latest">{probe?.loss != null ? probe.loss.toFixed(4) : 'no data'}</span>
            </div>
            <div className="tracked-sample-hbar-track" title={`Epoch ${epochNum}: loss ${(probe?.loss ?? 0).toFixed(4)}`}>
              <div className="tracked-sample-hbar-fill tracked-sample-bar-loss" style={{ width: `${lossPct}%` }} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// --- Generated Samples View (GAN / Diffusion) ---

function GeneratedSamplesView({ progress }: { progress: EpochData[] }) {
  // Find epochs that have generated samples
  const epochsWithSamples = progress.filter(ep => ep.generatedSamples?.length);

  if (epochsWithSamples.length === 0) {
    return (
      <div className="dashboard-chart-placeholder">
        No generated samples yet. Samples are generated every 5 epochs — keep training.
      </div>
    );
  }

  return (
    <div className="generated-samples-view">
      {epochsWithSamples.map(ep => (
        <div key={ep.epoch} className="generated-samples-epoch">
          <div className="generated-samples-epoch-label">Epoch {ep.epoch}</div>
          <div className="generated-samples-grid">
            {ep.generatedSamples!.map((pixels, i) => (
              <GeneratedSampleImage key={i} pixels={pixels} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function GeneratedSampleImage({ pixels }: { pixels: number[][] | number[][][] }) {
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
  }, [pixels]);
  return <canvas ref={canvasRef} className="generated-sample-img" />;
}
