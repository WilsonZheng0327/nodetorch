// MSE Loss: predictions vs targets + computation breakdown + error map

import { useRef, useEffect } from 'react';
import type { MseLossTransformation } from '../types';
import { FeatureMapsGrid } from './shared';

export function MseLossViz({ t }: { t: MseLossTransformation }) {
  return (
    <div className="tfm-mse-loss">
      {/* Predictions vs Targets */}
      {(t.predsFmaps || t.targetsFmaps) && (
        <div className="tfm-before-after">
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">Predictions {t.predsShape ? `[${t.predsShape.join(', ')}]` : ''}</div>
            {t.predsFmaps && <FeatureMapsGrid data={t.predsFmaps} />}
          </div>
          <div className="tfm-ba-divider" />
          <div className="tfm-ba-pane">
            <div className="tfm-ba-label">Targets {t.targetsShape ? `[${t.targetsShape.join(', ')}]` : ''}</div>
            {t.targetsFmaps && <FeatureMapsGrid data={t.targetsFmaps} />}
          </div>
        </div>
      )}

      {/* Error map */}
      {t.errorMap && t.errorH && t.errorW && (
        <div className="tfm-section">
          <div className="tfm-section-title">Per-pixel Error (brighter = larger difference)</div>
          <ErrorMapCanvas pixels={t.errorMap} height={t.errorH} width={t.errorW} />
        </div>
      )}

      {/* Computation breakdown */}
      <div className="tfm-section">
        <div className="tfm-section-title">Calculation</div>
        <div className="tfm-mse-calc">
          {t.numElements != null && (
            <div className="tfm-mse-calc-step">
              <span className="tfm-mse-calc-desc">1. Compute difference for each element</span>
              <span className="tfm-mse-calc-detail">{t.numElements.toLocaleString()} elements</span>
            </div>
          )}
          {t.meanAbsError != null && (
            <div className="tfm-mse-calc-step">
              <span className="tfm-mse-calc-desc">2. Mean |error|</span>
              <span className="tfm-mse-calc-val">{fmtV(t.meanAbsError)}</span>
            </div>
          )}
          {t.maxAbsError != null && (
            <div className="tfm-mse-calc-step">
              <span className="tfm-mse-calc-desc">3. Max |error|</span>
              <span className="tfm-mse-calc-val">{fmtV(t.maxAbsError)}</span>
            </div>
          )}
          {t.sumSquared != null && t.numElements != null && (
            <div className="tfm-mse-calc-step">
              <span className="tfm-mse-calc-desc">4. Sum of squared errors</span>
              <span className="tfm-mse-calc-val">{fmtV(t.sumSquared)}</span>
            </div>
          )}
          {t.loss != null && t.numElements != null && (
            <div className="tfm-mse-calc-step tfm-mse-calc-result">
              <span className="tfm-mse-calc-desc">5. MSE = sum / {t.numElements.toLocaleString()}</span>
              <span className="tfm-mse-calc-val tfm-mse-calc-final">{t.loss.toFixed(6)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ErrorMapCanvas({ pixels, height, width }: { pixels: number[][]; height: number; width: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const v = pixels[y]?.[x] ?? 0;
        const t = v / 255;
        data.data[idx] = Math.round(Math.min(255, t * 2 * 255));
        data.data[idx + 1] = Math.round(Math.max(0, (t - 0.3) * 1.4 * 255));
        data.data[idx + 2] = Math.round(Math.max(0, (t - 0.7) * 3.3 * 255));
        data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels, height, width]);
  return <canvas ref={canvasRef} className="tfm-mse-error-canvas" />;
}

function fmtV(v: number): string {
  if (v === 0) return '0';
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toFixed(1);
  if (abs >= 1) return v.toFixed(4);
  if (abs >= 0.0001) return v.toFixed(6);
  return v.toExponential(2);
}
