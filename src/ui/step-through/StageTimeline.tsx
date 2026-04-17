// Horizontal scrolling strip of StageCards — the "timeline" of forward-pass stages.

import { useRef, useEffect } from 'react';
import type { Stage } from './types';
import { StageCard } from './StageCard';

interface Props {
  stages: Stage[];
  currentIdx: number;
  onSelect: (idx: number) => void;
}

export function StageTimeline({ stages, currentIdx, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to current stage
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const active = container.querySelector('.stage-card-active') as HTMLElement | null;
    if (active) {
      active.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [currentIdx]);

  return (
    <div className="stage-timeline" ref={containerRef}>
      {stages.map((stage, i) => (
        <div key={stage.stageId} className="stage-timeline-item">
          <StageCard stage={stage} active={i === currentIdx} onClick={() => onSelect(i)} />
          {i < stages.length - 1 && <div className="stage-timeline-arrow">→</div>}
        </div>
      ))}
    </div>
  );
}
