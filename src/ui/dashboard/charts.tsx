// Canvas charts for the training dashboard: the loss/accuracy line chart, the
// gradient-flow bars, and the per-class accuracy bars. Styles come from
// TrainingDashboard.css (imported by the container).

import { useRef, useEffect, useState } from 'react';

// --- Canvas line chart (loss / accuracy, with optional val + compare series) ---

interface ChartProps {
  data: number[];
  labels: number[];
  color: string;
  formatValue: (v: number) => string;
  selectedIndex?: number | null;
  valData?: (number | null | undefined)[];
  valColor?: string;
  compareData?: (number | null | undefined)[];
  compareColor?: string;
  compareLabel?: string;
}

export function Chart({ data: rawData, labels, color, formatValue, selectedIndex, valData, valColor, compareData, compareColor, compareLabel }: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Track canvas client size so we re-render on resize (keeps bitmap crisp)
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const update = () => setSize({ w: canvas.clientWidth, h: canvas.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rawData.length === 0 || size.w === 0) return;

    // Guard against NaN/null/Infinity
    const data = rawData.map((v) => (v == null || !isFinite(v)) ? 0 : v);
    // Sanitize valData (preserve null = "no val for this epoch")
    const valClean = valData?.map((v) => (v != null && isFinite(v) ? v : null));
    const compareClean = compareData?.map((v) => (v != null && isFinite(v) ? v : null));

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    // Use clientWidth/Height (excludes padding) for accurate bitmap sizing.
    const w = size.w;
    const h = size.h;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const pad = { top: 40, right: 12, bottom: 28, left: 50 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    // Compute min/max across all series
    const valValuesForRange = (valClean?.filter((v): v is number => v != null)) ?? [];
    const compareValuesForRange = (compareClean?.filter((v): v is number => v != null)) ?? [];
    const allValues = [...data, ...valValuesForRange, ...compareValuesForRange];
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const range = max - min || 1;

    // Y axis labels
    ctx.fillStyle = '#6c7086';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = min + (range * i) / 4;
      const y = pad.top + plotH - (plotH * i) / 4;
      ctx.fillText(formatValue(val), pad.left - 6, y + 3);

      ctx.strokeStyle = '#313244';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // X axis labels
    ctx.textAlign = 'center';
    ctx.fillStyle = '#6c7086';
    const step = Math.max(1, Math.floor(data.length / 6));
    for (let i = 0; i < data.length; i += step) {
      const x = pad.left + (plotW * i) / Math.max(data.length - 1, 1);
      ctx.fillText(String(labels[i]), x, h - 4);
    }

    // Data line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (plotW * i) / Math.max(data.length - 1, 1);
      const y = pad.top + plotH - (plotH * (data[i] - min)) / range;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Data points
    ctx.fillStyle = color;
    for (let i = 0; i < data.length; i++) {
      const x = pad.left + (plotW * i) / Math.max(data.length - 1, 1);
      const y = pad.top + plotH - (plotH * (data[i] - min)) / range;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Validation line (dashed) if provided
    if (valClean) {
      const vColor = valColor ?? '#fab387';
      ctx.strokeStyle = vColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < valClean.length; i++) {
        const v = valClean[i];
        if (v == null) {
          started = false;
          continue;
        }
        const x = pad.left + (plotW * i) / Math.max(valClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // Val points
      ctx.fillStyle = vColor;
      for (let i = 0; i < valClean.length; i++) {
        const v = valClean[i];
        if (v == null) continue;
        const x = pad.left + (plotW * i) / Math.max(valClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Compare line (dotted, purple by default)
    if (compareClean && compareClean.length > 0) {
      const cColor = compareColor ?? '#cba6f7';
      ctx.strokeStyle = cColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([1, 3]);
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < compareClean.length; i++) {
        const v = compareClean[i];
        if (v == null) {
          started = false;
          continue;
        }
        const x = pad.left + (plotW * i) / Math.max(compareClean.length - 1, 1);
        const y = pad.top + plotH - (plotH * (v - min)) / range;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // Legend (if val line present). Sits in the top margin with breathing room above.
    if (valClean) {
      const legendY = 18;       // text baseline
      const legendRectY = 12;   // line sample top
      ctx.font = '11px Inter, system-ui, sans-serif';
      ctx.textAlign = 'left';
      // Train legend
      ctx.fillStyle = color;
      ctx.fillRect(pad.left + 4, legendRectY, 10, 2);
      ctx.fillStyle = '#a6adc8';
      ctx.fillText('train', pad.left + 18, legendY);
      // Val legend
      ctx.strokeStyle = valColor ?? '#fab387';
      ctx.setLineDash([3, 2]);
      ctx.beginPath();
      ctx.moveTo(pad.left + 60, legendRectY + 1);
      ctx.lineTo(pad.left + 70, legendRectY + 1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#a6adc8';
      ctx.fillText('val', pad.left + 74, legendY);
      // Compare legend
      if (compareClean && compareClean.length > 0) {
        ctx.strokeStyle = compareColor ?? '#cba6f7';
        ctx.setLineDash([1, 3]);
        ctx.beginPath();
        ctx.moveTo(pad.left + 110, legendRectY + 1);
        ctx.lineTo(pad.left + 120, legendRectY + 1);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#cba6f7';
        const labelText = compareLabel && compareLabel.length < 40 ? compareLabel : 'compare';
        ctx.fillText(labelText, pad.left + 124, legendY);
      }
    }

    // Epoch marker: a dotted vertical line + dot at the epoch in view. When
    // scrubbing history it's blue at the selected epoch; when viewing the latest
    // epoch (selectedIndex == null) it's orange at the final point to signal
    // "current / latest".
    const isCurrentEpoch = selectedIndex == null;
    const markerIndex = isCurrentEpoch
      ? data.length - 1
      : (selectedIndex >= 0 && selectedIndex < data.length ? selectedIndex : -1);
    if (markerIndex >= 0) {
      const markerColor = isCurrentEpoch ? '#fe640b' : '#89b4fa';
      const sx = pad.left + (plotW * markerIndex) / Math.max(data.length - 1, 1);
      const sy = pad.top + plotH - (plotH * (data[markerIndex] - min)) / range;
      // Vertical line
      ctx.strokeStyle = markerColor;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx, pad.top);
      ctx.lineTo(sx, pad.top + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      // Highlight dot
      ctx.fillStyle = markerColor;
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fill();
      // Value label
      ctx.fillStyle = '#cdd6f4';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(formatValue(data[markerIndex]), sx, sy - 10);
    }
  }, [rawData, labels, color, formatValue, selectedIndex, valData, valColor, compareData, compareColor, compareLabel, size]);

  return <canvas ref={canvasRef} className="dashboard-chart" />;
}

// --- Gradient flow horizontal bar chart ---

export function GradientFlowChart({ data }: { data: { name: string; norm: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const barH = 22;
    const gap = 2;
    const labelW = 160;
    const pad = { top: 8, right: 80, bottom: 8 };
    const totalH = pad.top + data.length * (barH + gap) + pad.bottom;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = totalH * dpr;
    canvas.style.height = `${totalH}px`;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const plotW = w - labelW - pad.right;
    const maxNorm = Math.max(...data.map((d) => d.norm), 1e-8);

    ctx.clearRect(0, 0, w, totalH);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + gap);
      const barW = (data[i].norm / maxNorm) * plotW;

      // Label (truncated if too long for labelW)
      ctx.fillStyle = '#a6adc8';
      ctx.font = '12px Inter, system-ui, sans-serif';
      ctx.textAlign = 'right';
      let label = data[i].name;
      if (ctx.measureText(label).width > labelW - 8) {
        while (label.length > 3 && ctx.measureText(label + '…').width > labelW - 8) {
          label = label.slice(0, -1);
        }
        label = label + '…';
      }
      ctx.fillText(label, labelW - 6, y + barH / 2 + 3);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const ratio = data[i].norm / maxNorm;
      const r = Math.round(Math.min(255, ratio * 2 * 255));
      const g = Math.round(Math.min(255, (1 - ratio) * 2 * 255));
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      ctx.fillStyle = '#6c7086';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(data[i].norm.toExponential(1), labelW + barW + 4, y + barH / 2 + 3);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No gradient data yet</div>;
  }

  return (
    <div className="dashboard-gradflow-scroll">
      <div className="dashboard-explainer">
        Gradient magnitude (L2 norm) per layer after backward pass. Bars near zero at early
        layers can indicate vanishing gradients; very large bars can indicate exploding
        gradients. Healthy training usually shows gradients within a few orders of magnitude.
      </div>
      <canvas ref={canvasRef} className="dashboard-chart dashboard-chart-tall" />
    </div>
  );
}

// --- Per-class accuracy bar chart (all classes, sorted worst-first, scrollable) ---

export function PerClassChart({ data }: { data: { cls: number; accuracy: number }[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const barH = 24;
    const gap = 3;
    const labelW = 48;
    const pad = { top: 8, right: 60, bottom: 8 };
    const totalH = pad.top + data.length * (barH + gap) + pad.bottom;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = totalH * dpr;
    canvas.style.height = `${totalH}px`;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const plotW = w - labelW - pad.right;

    ctx.clearRect(0, 0, w, totalH);

    for (let i = 0; i < data.length; i++) {
      const y = pad.top + i * (barH + gap);
      const barW = data[i].accuracy * plotW;

      ctx.fillStyle = '#a6adc8';
      ctx.font = '13px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(data[i].cls), labelW - 6, y + barH / 2 + 4);

      ctx.fillStyle = '#313244';
      ctx.fillRect(labelW, y, plotW, barH);

      const acc = data[i].accuracy;
      const r = Math.round((1 - acc) * 230);
      const g = Math.round(acc * 200);
      ctx.fillStyle = `rgba(${r}, ${g}, 100, 0.7)`;
      ctx.fillRect(labelW, y, barW, barH);

      ctx.fillStyle = '#cdd6f4';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${(acc * 100).toFixed(1)}%`, labelW + barW + 6, y + barH / 2 + 4);
    }
  }, [data]);

  if (data.length === 0) {
    return <div className="dashboard-chart-placeholder">No per-class data yet</div>;
  }

  return (
    <div className="dashboard-perclass-scroll">
      <canvas ref={canvasRef} className="dashboard-chart dashboard-chart-tall" />
    </div>
  );
}
