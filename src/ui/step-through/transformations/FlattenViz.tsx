// Flatten transformation:
// - Shape calculation: [C × H × W] = N values
// - Input as 2D image
// - Flattened result as a scrollable horizontal pixel strip

import { useRef, useEffect } from 'react';
import type { FlattenTransformation } from '../types';
import { FeatureMapsGrid } from './shared';

export function FlattenViz({ t }: { t: FlattenTransformation }) {
  const dims = t.inputShape;
  const product = dims.reduce((a, b) => a * b, 1);
  const dimStr = dims.join(' \u00d7 ');

  return (
    <div className="tfm-flatten">
      {/* Shape calculation */}
      <div className="tfm-flatten-calc">
        <span className="tfm-flatten-calc-before">[{dimStr}]</span>
        <span className="tfm-flatten-calc-arrow">&rarr;</span>
        <span className="tfm-flatten-calc-after">[{product}]</span>
        {dims.length > 1 && (
          <span className="tfm-flatten-calc-explain">{dimStr} = {product}</span>
        )}
      </div>

      {/* Before: 2D image */}
      <div className="tfm-flatten-before">
        <div className="tfm-ba-label">Before</div>
        {t.inputMaps ? (
          <FeatureMapsGrid data={t.inputMaps} />
        ) : (
          <div className="tfm-flatten-shape-block">
            {dims.map((d, i) => (
              <span key={i}>
                {i > 0 && <span className="tfm-flatten-x">&times;</span>}
                <span className="tfm-flatten-dim">{d}</span>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* After: flattened pixel strip */}
      <div className="tfm-flatten-after">
        <div className="tfm-ba-label">After &middot; {t.outputLength} values unrolled into a single row</div>
        {t.flatPixels && t.flatPixels.length > 0 && (
          <FlatPixelStrip pixels={t.flatPixels} />
        )}
      </div>
    </div>
  );
}

/** Renders pixel values as a scrollable horizontal strip.
 *  Each pixel is rendered as a small square so proportions are balanced. */
function FlatPixelStrip({ pixels }: { pixels: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const pixelSize = 4; // each value = 4×4 px square
  const stripH = 32;   // strip height in px

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const w = pixels.length * pixelSize;
    const h = stripH;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    for (let i = 0; i < pixels.length; i++) {
      const v = pixels[i];
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(i * pixelSize, 0, pixelSize, h);
    }
  }, [pixels]);

  return (
    <div className="tfm-flatten-strip-wrap">
      <canvas ref={canvasRef} className="tfm-flatten-strip-canvas" />
      <div className="tfm-flatten-strip-labels">
        <span>index 0</span>
        <span>&rarr;</span>
      </div>
    </div>
  );
}
