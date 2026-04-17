// Keyboard shortcuts help modal — opened by pressing "?".

import { useEffect } from 'react';
import './ShortcutsHelp.css';

interface Props {
  open: boolean;
  onClose: () => void;
}

interface Shortcut {
  keys: string;
  description: string;
}

const SHORTCUTS: { category: string; items: Shortcut[] }[] = [
  {
    category: 'Canvas',
    items: [
      { keys: 'Tab', description: 'Toggle node palette' },
      { keys: 'W / A / S / D', description: 'Pan camera (also arrow keys)' },
      { keys: 'Scroll', description: 'Zoom in / out' },
      { keys: 'Drag blank area', description: 'Pan camera' },
      { keys: '?', description: 'Show this help' },
    ],
  },
  {
    category: 'Nodes',
    items: [
      { keys: 'Click', description: 'Select node (opens inspector)' },
      { keys: 'Shift + Click', description: 'Add to selection (multi-select)' },
      { keys: 'Ctrl + drag', description: 'Box-select multiple nodes' },
      { keys: 'Double-click (subgraph)', description: 'Enter subgraph' },
      { keys: 'Drag from palette', description: 'Add new node' },
      { keys: 'Drag node', description: 'Move node' },
      { keys: 'Delete / Backspace', description: 'Delete selected node(s)' },
    ],
  },
  {
    category: 'Edges',
    items: [
      { keys: 'Drag port to port', description: 'Connect nodes' },
      { keys: 'Right-click edge', description: 'Delete edge' },
    ],
  },
  {
    category: 'Editing',
    items: [
      { keys: 'Ctrl + Z', description: 'Undo' },
      { keys: 'Ctrl + Shift + Z', description: 'Redo' },
      { keys: 'Ctrl + C', description: 'Copy selected nodes' },
      { keys: 'Ctrl + V', description: 'Paste copied nodes' },
    ],
  },
];


export function ShortcutsHelp({ open, onClose }: Props) {
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape' && open) onClose();
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="shortcuts-overlay" onClick={onClose}>
      <div className="shortcuts-modal" onClick={(e) => e.stopPropagation()}>
        <div className="shortcuts-header">
          <span>Keyboard Shortcuts</span>
          <button className="shortcuts-close" onClick={onClose}>&times;</button>
        </div>
        <div className="shortcuts-body">
          {SHORTCUTS.map((cat) => (
            <div key={cat.category} className="shortcuts-category">
              <div className="shortcuts-category-title">{cat.category}</div>
              {cat.items.map((s, i) => (
                <div key={i} className="shortcuts-row">
                  <span className="shortcuts-keys">{s.keys}</span>
                  <span className="shortcuts-desc">{s.description}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
