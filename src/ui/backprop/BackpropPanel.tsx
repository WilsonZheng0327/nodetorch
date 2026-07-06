// BackpropPanel — fullscreen overlay, a shell hosting swappable "pages":
//   • Over training — the main view: watch the model's predictions on a few fixed
//     examples improve epoch by epoch (replays the last training run).
//   • Per layer — a drill-in that steps through the backward pass layer by layer
//     (loss seed, the update, wiggle-a-weight).
// Pages are a plain list, so adding/removing one is a one-line change.

import { useState } from 'react';
import type { EpochData } from '../dashboard/types';
import { EpochPlayback } from './EpochPlayback';
import { PerLayerView } from './PerLayerView';
import '../step-through/StepThroughPanel.css';
import './BackpropPanel.css';

interface Props {
  open: boolean;
  graphJson: string;
  trainingProgress: EpochData[];
  onClose: () => void;
}

const PAGES = [
  { id: 'playback', label: 'Over training' },
  { id: 'perlayer', label: 'Per layer' },
] as const;
type PageId = (typeof PAGES)[number]['id'];

export function BackpropPanel({ open, graphJson, trainingProgress, onClose }: Props) {
  const [page, setPage] = useState<PageId>('playback');
  if (!open) return null;

  return (
    <div className="step-through-panel">
      <div className="step-through-header">
        <span className="step-through-title">Backprop</span>
        <div className="backprop-page-tabs" role="tablist">
          {PAGES.map((p) => (
            <button
              key={p.id}
              role="tab"
              aria-selected={page === p.id}
              className={`backprop-page-tab ${page === p.id ? 'backprop-page-tab-active' : ''}`}
              onClick={() => setPage(p.id)}
            >
              {p.label}
            </button>
          ))}
        </div>
        <div className="step-through-header-actions">
          <button className="step-through-close" onClick={onClose}>
            &times;
          </button>
        </div>
      </div>

      {page === 'playback' && <EpochPlayback progress={trainingProgress} />}
      {page === 'perlayer' && <PerLayerView graphJson={graphJson} />}
    </div>
  );
}
