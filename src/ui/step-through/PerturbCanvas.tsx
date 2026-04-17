// PerturbCanvas — lets the user draw a mask on top of the sample image.
// The mask is a 2D array matching the image H×W. Value 1 = mask (zero pixel).
// Parent component calls onMaskChange whenever the user paints.

import { useRef, useEffect, useState } from 'react';

interface Props {
  pixels: number[][] | number[][][];
  channels: number;
  mask: number[][] | null;
  onMaskChange: (mask: number[][]) => void;
  displaySize?: number;  // CSS size (px)
  brushRadius?: number;  // in image-coordinate pixels
}

export function PerturbCanvas({ pixels, channels, mask, onMaskChange, displaySize = 160, brushRadius = 2 }: Props) {
  const imgCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const H = pixels.length;
  const W = Array.isArray(pixels[0]) && Array.isArray((pixels[0] as number[][])[0])
    ? (pixels[0] as number[][]).length
    : (pixels[0] as number[]).length;

  // Initialize mask if needed
  useEffect(() => {
    if (!mask || mask.length !== H || mask[0]?.length !== W) {
      const empty: number[][] = [];
      for (let y = 0; y < H; y++) {
        empty.push(new Array(W).fill(0));
      }
      onMaskChange(empty);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [H, W]);

  // Render the base image
  useEffect(() => {
    const canvas = imgCanvasRef.current;
    if (!canvas) return;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const data = ctx.createImageData(W, H);
    const isRGB = channels >= 3;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
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
  }, [pixels, channels, W, H]);

  // Render the mask overlay
  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas || !mask) return;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);
    const data = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const idx = (y * W + x) * 4;
        if (mask[y] && mask[y][x]) {
          data.data[idx] = 243;     // pink
          data.data[idx + 1] = 139;
          data.data[idx + 2] = 168;
          data.data[idx + 3] = 200;
        }
      }
    }
    ctx.putImageData(data, 0, 0);
  }, [mask, W, H]);

  // Mouse → image coordinate conversion
  function eventToImageCoord(e: React.PointerEvent) {
    const canvas = maskCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;
    const x = Math.floor((cssX / rect.width) * W);
    const y = Math.floor((cssY / rect.height) * H);
    return { x, y };
  }

  function paint(x: number, y: number) {
    if (!mask) return;
    const next = mask.map((row) => [...row]);
    for (let dy = -brushRadius; dy <= brushRadius; dy++) {
      for (let dx = -brushRadius; dx <= brushRadius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
          if (dx * dx + dy * dy <= brushRadius * brushRadius) {
            next[ny][nx] = 1;
          }
        }
      }
    }
    onMaskChange(next);
  }

  function handlePointerDown(e: React.PointerEvent) {
    const canvas = maskCanvasRef.current;
    if (canvas) canvas.setPointerCapture(e.pointerId);
    setIsDrawing(true);
    const c = eventToImageCoord(e);
    if (c) paint(c.x, c.y);
  }

  function handlePointerMove(e: React.PointerEvent) {
    if (!isDrawing) return;
    const c = eventToImageCoord(e);
    if (c) paint(c.x, c.y);
  }

  function handlePointerUp() {
    setIsDrawing(false);
  }

  return (
    <div className="perturb-canvas-wrap" style={{ width: displaySize, height: displaySize }}>
      <canvas
        ref={imgCanvasRef}
        className="perturb-canvas-img"
        style={{ width: displaySize, height: displaySize }}
      />
      <canvas
        ref={maskCanvasRef}
        className="perturb-canvas-mask"
        style={{ width: displaySize, height: displaySize, cursor: 'crosshair' }}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      />
    </div>
  );
}
