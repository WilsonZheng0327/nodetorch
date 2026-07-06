// SampleMenu — the small "change" control that lives on the data node's timeline
// card (both forward and backward panels). Opens a popover to draw a new random
// sample, or, when the dataset has class labels, pick a sample by class.

import { useState, useRef, useEffect } from 'react';
import { Shuffle } from 'lucide-react';

export interface DataControls {
  onNewSample: () => void;
  onPickLabel?: (label: number) => void;
  classNames?: string[] | null;
  currentLabel?: number | null;
  loading?: boolean;
}

export function SampleMenu({
  onNewSample,
  onPickLabel,
  classNames,
  currentLabel,
  loading,
}: DataControls) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 });
  const ref = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const toggle = () => {
    const r = btnRef.current?.getBoundingClientRect();
    if (r) {
      const width = 224;
      setPos({ top: r.bottom + 4, left: Math.min(r.left, window.innerWidth - width - 12) });
    }
    setOpen((o) => !o);
  };

  const hasClasses = !!classNames && classNames.length > 0 && !!onPickLabel;

  return (
    <div className="sample-menu" ref={ref}>
      <button
        ref={btnRef}
        className="sample-menu-btn"
        onClick={toggle}
        disabled={loading}
        title="Change the input sample"
        aria-label="Change the input sample"
      >
        <Shuffle size={13} />
      </button>
      {open && (
        // position: fixed so the popup escapes the timeline's scroll clipping
        <div className="sample-menu-pop" style={{ top: pos.top, left: pos.left }}>
          <button
            className="sample-menu-item"
            onClick={() => {
              onNewSample();
              setOpen(false);
            }}
          >
            ⟳ New random sample
          </button>
          {hasClasses && (
            <>
              <div className="sample-menu-heading">By class</div>
              <div className="sample-menu-classes">
                {classNames!.map((name, i) => (
                  <button
                    key={i}
                    className={`sample-menu-class ${currentLabel === i ? 'sample-menu-class-active' : ''}`}
                    onClick={() => {
                      onPickLabel!(i);
                      setOpen(false);
                    }}
                    title={name}
                  >
                    {name}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
