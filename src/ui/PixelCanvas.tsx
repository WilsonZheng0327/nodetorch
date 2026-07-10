// PixelCanvas — draws a pixel array onto a <canvas>. Used everywhere a sample /
// feature-map / generated image is previewed. `pixels` is either [y][x] grayscale
// (0-255) or [y][x][channel] RGB; `channels >= 3` (or a nested first row) → RGB.

import { useRef, useEffect } from 'react';

export function PixelCanvas({
  pixels,
  channels = 1,
  className,
}: {
  pixels: number[][] | number[][][];
  channels?: number;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || pixels.length === 0) return;
    const h = pixels.length;
    const firstRow = pixels[0] as number[] | number[][];
    const isRGB = channels >= 3 || Array.isArray((firstRow as number[][])[0]);
    const w = isRGB ? (firstRow as number[][]).length : (firstRow as number[]).length;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(w, h);
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
  return <canvas ref={ref} className={className} />;
}
