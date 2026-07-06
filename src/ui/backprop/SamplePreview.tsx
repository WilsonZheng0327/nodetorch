// SamplePreview — the input the backward pass is computing gradients for. Mirrors
// the forward step-through's sample header exactly (same markup + CSS classes) so
// the two panels look identical.

import { useRef, useEffect } from 'react';
import type { BackwardSample } from './types';

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
        <SampleImage pixels={sample.imagePixels} channels={sample.imageChannels ?? 1} />
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

function SampleImage({
  pixels,
  channels,
}: {
  pixels: number[][] | number[][][];
  channels: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length;
    const firstRow = pixels[0] as number[] | number[][];
    const w = Array.isArray((firstRow as number[][])[0])
      ? (firstRow as number[][]).length
      : (firstRow as number[]).length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
    const isRGB = channels >= 3;
    for (let y = 0; y < h; y++)
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
    ctx.putImageData(data, 0, 0);
  }, [pixels, channels]);
  return <canvas ref={canvasRef} className="step-through-sample-img" />;
}
