// Samples-tab views: tracked classification/reconstruction samples, generated
// text (autoregressive), and generated images (GAN / diffusion). Styles come
// from TrainingDashboard.css.

import type { EpochData, TrackedSampleProbe } from './types';
import { PixelCanvas } from '../PixelCanvas';

// --- Generated text (autoregressive) ---

export function GeneratedTextView({ progress }: { progress: EpochData[] }) {
  const samples = progress
    .filter((ep) => ep.generatedText)
    .map((ep) => ({ epoch: ep.epoch, text: ep.generatedText! }));

  if (samples.length === 0) {
    return (
      <div className="dashboard-chart-placeholder">
        No generated samples yet — text is generated periodically during training
      </div>
    );
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

export function TrackedSamplesView({
  progress,
  selectedEpoch,
}: {
  progress: EpochData[];
  selectedEpoch: number | null;
}) {
  // For GAN/diffusion, show generated samples instead of tracked samples
  const isGenerative = progress.some((ep) => ep.generatedSamples?.length);
  if (isGenerative) {
    return <GeneratedSamplesView progress={progress} />;
  }

  if (progress.length === 0 || !progress[0].trackedSamples?.length) {
    return (
      <div className="dashboard-chart-placeholder">No tracked samples — train to see results</div>
    );
  }

  // Get sample info from the first epoch (images don't change)
  const firstProbes = progress[0].trackedSamples!;
  const classNames = progress[0].classNames ?? null;

  return (
    <div className="tracked-samples-view">
      {firstProbes.map((sample, sIdx) => (
        <TrackedSampleRow
          key={sample.idx}
          sampleIdx={sIdx}
          sample={sample}
          probes={progress.map((ep) => ep.trackedSamples?.[sIdx])}
          epochs={progress.map((ep) => ep.epoch)}
          selectedEpoch={selectedEpoch}
          classNames={classNames}
        />
      ))}
    </div>
  );
}

function TrackedSampleRow({
  sample,
  probes,
  epochs,
  selectedEpoch,
  classNames,
}: {
  sampleIdx: number;
  sample: TrackedSampleProbe;
  probes: (TrackedSampleProbe | undefined)[];
  epochs: number[];
  selectedEpoch: number | null;
  classNames: string[] | null;
}) {
  const clsLabel = (c: number | null | undefined): string =>
    c == null ? '?' : classNames && c < classNames.length ? classNames[c] : String(c);
  // The epoch in view: the slider selection (1-based) or the latest.
  const selIdx =
    selectedEpoch != null
      ? Math.min(Math.max(selectedEpoch - 1, 0), probes.length - 1)
      : probes.length - 1;
  const probe = probes[selIdx];
  const epochNum = epochs[selIdx];
  // Classification vs reconstruction is a fixed property of the model.
  const isClassification = (probes[probes.length - 1] ?? probe)?.probabilities != null;

  const correct = probe?.predictedClass === sample.label;
  const conf = (probe?.confidence ?? 0) * 100;
  const maxLoss = Math.max(...probes.filter(Boolean).map((p) => p!.loss ?? 0), 0.01);
  const lossPct = ((probe?.loss ?? 0) / maxLoss) * 100;

  return (
    <div className="tracked-sample-row">
      {/* Thumbnail */}
      <div className="tracked-sample-thumb">
        {sample.imagePixels ? (
          <PixelCanvas pixels={sample.imagePixels} className="tracked-sample-img" />
        ) : (
          <div className="tracked-sample-no-img">#{sample.idx}</div>
        )}
        <div className="tracked-sample-label">Label: {clsLabel(sample.label)}</div>
      </div>

      {/* Selected-epoch prediction — one horizontal bar that scrubs with the slider */}
      <div className="tracked-sample-timeline">
        {isClassification ? (
          <>
            <div className="tracked-sample-timeline-header">
              <span>Confidence (epoch {epochNum})</span>
              <span className="tracked-sample-latest">
                {probe?.predictedClass != null ? (
                  <>
                    Predicted: {clsLabel(probe.predictedClass)} ({conf.toFixed(1)}%)
                  </>
                ) : (
                  'no data'
                )}
              </span>
            </div>
            <div
              className="tracked-sample-hbar-track"
              title={`Epoch ${epochNum}: class ${clsLabel(probe?.predictedClass)} (${conf.toFixed(1)}%)`}
            >
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
              <span className="tracked-sample-latest">
                {probe?.loss != null ? probe.loss.toFixed(4) : 'no data'}
              </span>
            </div>
            <div
              className="tracked-sample-hbar-track"
              title={`Epoch ${epochNum}: loss ${(probe?.loss ?? 0).toFixed(4)}`}
            >
              <div
                className="tracked-sample-hbar-fill tracked-sample-bar-loss"
                style={{ width: `${lossPct}%` }}
              />
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
  const epochsWithSamples = progress.filter((ep) => ep.generatedSamples?.length);

  if (epochsWithSamples.length === 0) {
    return (
      <div className="dashboard-chart-placeholder">
        No generated samples yet. Samples are generated every 5 epochs — keep training.
      </div>
    );
  }

  return (
    <div className="generated-samples-view">
      {epochsWithSamples.map((ep) => (
        <div key={ep.epoch} className="generated-samples-epoch">
          <div className="generated-samples-epoch-label">Epoch {ep.epoch}</div>
          <div className="generated-samples-grid">
            {ep.generatedSamples!.map((pixels, i) => (
              <PixelCanvas key={i} pixels={pixels} className="generated-sample-img" />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
