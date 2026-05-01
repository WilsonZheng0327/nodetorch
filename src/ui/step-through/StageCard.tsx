// Small card shown in the horizontal timeline, one per stage.
// Only shows feature map previews for layers that produce spatial data.
// Other layers just show an enlarged name and shape.

import { useRef, useEffect } from 'react';
import type { Stage, Transformation, FeatureMaps } from './types';
import { compactShape } from './insights';

interface Props {
  stage: Stage;
  active: boolean;
  onClick: () => void;
}

export function StageCard({ stage, active, onClick }: Props) {
  const title = stage.blockName ? `${stage.blockName} > ${stage.displayName}` : stage.displayName;
  const fmaps = stage.transformation ? extractOutputFeatureMaps(stage.transformation) : undefined;
  const hasPreview = fmaps && fmaps.maps.length > 0;

  return (
    <button
      className={`stage-card ${active ? 'stage-card-active' : ''} ${stage.depth > 0 ? 'stage-card-nested' : ''} ${!hasPreview ? 'stage-card-no-preview' : ''}`}
      onClick={onClick}
      title={title}
    >
      {stage.blockName && <div className="stage-card-block">{stage.blockName}</div>}
      <div className={`stage-card-name ${!hasPreview ? 'stage-card-name-large' : ''}`}>{stage.displayName}</div>
      {hasPreview && (
        <div className="stage-card-preview">
          <MiniFeatureMap pixels={fmaps!.maps[0]} />
        </div>
      )}
      <div className="stage-card-shape">{compactShape(stage.outputShape)}</div>
    </button>
  );
}

/** Only extract feature maps for spatial layers — conv, pool, upsample, data, dropout. */
function extractOutputFeatureMaps(t: Transformation): FeatureMaps | undefined {
  switch (t.type) {
    case 'conv2d': return t.output;
    case 'pool': return (t.output.height > 2 && t.output.width > 2) ? t.output : undefined;
    case 'upsample': return t.output;
    case 'data': return t.featureMaps;
    case 'dropout': return t.outputMaps;
    case 'activation': return t.outputMaps;
    default: return undefined;
  }
}

function MiniFeatureMap({ pixels }: { pixels: number[][] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length;
    const w = pixels[0].length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = (y * w + x) * 4;
        const v = pixels[y][x];
        data.data[idx] = v; data.data[idx + 1] = v; data.data[idx + 2] = v; data.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [pixels]);
  return <canvas ref={canvasRef} className="stage-card-canvas" />;
}
