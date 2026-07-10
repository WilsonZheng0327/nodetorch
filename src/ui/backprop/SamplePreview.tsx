// SamplePreview — the input the backward pass is computing gradients for. Mirrors
// the forward step-through's sample header exactly (same markup + CSS classes) so
// the two panels look identical.

import type { BackwardSample } from './types';
import { PixelCanvas } from '../PixelCanvas';

export function SamplePreview({
  sample,
  onNewSample,
  loading,
}: {
  sample: BackwardSample;
  onNewSample?: () => void;
  loading?: boolean;
}) {
  const label = sample.actualLabel;
  const labelValue =
    label != null && sample.classNames && label < sample.classNames.length
      ? `${sample.classNames[label]} (${label})`
      : label != null
        ? String(label)
        : null;

  return (
    <div className="step-through-sample">
      {sample.imagePixels && (
        <PixelCanvas
          pixels={sample.imagePixels}
          channels={sample.imageChannels ?? 1}
          className="step-through-sample-img"
        />
      )}
      <div className="step-through-sample-info">
        {labelValue != null && (
          <div>
            <span className="step-through-sample-label">Label: </span>
            <span className="step-through-sample-value">{labelValue}</span>
          </div>
        )}
        {sample.datasetType && (
          <div className="step-through-sample-dataset">{sample.datasetType}</div>
        )}
        {sample.sampleText && <div className="step-through-sample-text">{sample.sampleText}</div>}
      </div>
      {onNewSample && (
        <div className="step-through-sample-actions">
          <button className="step-through-btn" onClick={onNewSample} disabled={loading}>
            {loading && <span className="step-through-spinner" aria-hidden="true" />}
            New sample
          </button>
        </div>
      )}
    </div>
  );
}
