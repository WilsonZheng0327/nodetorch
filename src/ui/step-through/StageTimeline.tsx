// Horizontal scrolling strip of StageCards — the "timeline" of forward-pass stages.

import { useRef, useEffect } from 'react';
import type { Stage } from './types';
import { StageCard } from './StageCard';
import { SampleMenu, type DataControls } from './SampleMenu';

interface Props {
  stages: Stage[];
  currentIdx: number;
  onSelect: (idx: number) => void;
  direction?: 'forward' | 'backward';
  /** Horizontal alignment. 'right' hugs the output end (used by the backward view). */
  align?: 'left' | 'right';
  /** When set, the data-node card gets a "change" button to pick a new sample. */
  dataControls?: DataControls;
}

/** The data/input node — its card hosts the sample-change control. */
const isDataStage = (stage: Stage): boolean => !!stage.nodeType?.startsWith('data');

export function StageTimeline({
  stages,
  currentIdx,
  onSelect,
  direction = 'forward',
  align = 'left',
  dataControls,
}: Props) {
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

    const withMenu = dataControls && isDataStage(stage);
    items.push(
      <div key={stage.stageId} className="stage-timeline-item">
        <div className={`stage-card-wrap ${withMenu ? 'stage-card-wrap-data' : ''}`}>
          {withMenu && <SampleMenu {...dataControls} />}
          <StageCard stage={stage} active={i === currentIdx} onClick={() => onSelect(i)} />
        </div>
        {i < stages.length - 1 && (
          <div className="stage-timeline-arrow">{direction === 'backward' ? '←' : '→'}</div>
        )}
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
    <div
      className={`stage-timeline ${align === 'right' ? 'stage-timeline-right' : ''}`}
      ref={containerRef}
    >
      {items}
    </div>
  );
}
