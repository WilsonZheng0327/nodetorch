// Horizontal scrolling strip of StageCards — the "timeline" of forward-pass stages.

import { useRef, useEffect } from 'react';
import type { Stage } from './types';
import { StageCard } from './StageCard';

interface Props {
  stages: Stage[];
  currentIdx: number;
  onSelect: (idx: number) => void;
  direction?: 'forward' | 'backward';
}

export function StageTimeline({ stages, currentIdx, onSelect, direction = 'forward' }: Props) {
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

  // Compute "block boundary" markers — when the containing block changes between consecutive stages
  const items: React.ReactNode[] = [];
  let prevBlock: string | null = null;

  stages.forEach((stage, i) => {
    const currentBlock = stage.blockName ?? null;

    // Insert a divider when entering/leaving a block
    if (currentBlock !== prevBlock) {
      if (currentBlock) {
        items.push(
          <div key={`enter-${stage.stageId}`} className="stage-timeline-block-enter">
            <span className="stage-timeline-block-label">{currentBlock}</span>
            <span className="stage-timeline-block-bracket">⌐</span>
          </div>,
        );
      } else if (prevBlock) {
        items.push(
          <div key={`exit-${stage.stageId}`} className="stage-timeline-block-exit">
            <span className="stage-timeline-block-bracket">¬</span>
          </div>,
        );
      }
      prevBlock = currentBlock;
    }

    items.push(
      <div key={stage.stageId} className="stage-timeline-item">
        <StageCard stage={stage} active={i === currentIdx} onClick={() => onSelect(i)} />
        {i < stages.length - 1 && <div className="stage-timeline-arrow">{direction === 'backward' ? '←' : '→'}</div>}
      </div>,
    );
  });

  // Closing bracket if we ended inside a block
  if (prevBlock) {
    items.push(
      <div key="exit-end" className="stage-timeline-block-exit">
        <span className="stage-timeline-block-bracket">¬</span>
      </div>,
    );
  }

  return (
    <div className="stage-timeline" ref={containerRef}>
      {items}
    </div>
  );
}
