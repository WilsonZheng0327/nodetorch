// LeftRail — docked left panel hosting the node palette and the property
// inspector behind two tabs. Selecting a node auto-switches to Inspector;
// clearing the selection falls back to Nodes. Either tab can be pinned
// manually. The Tab key collapses/expands the whole rail.

import './LeftRail.css';

import { useEffect, useState } from 'react';
import { Blocks, SlidersHorizontal, ChevronLeft, ChevronRight } from 'lucide-react';
import { NodePalette } from '../NodePalette';
import { PropertyInspector } from '../inspector/PropertyInspector';
import { tutorialEvent } from '../tutorial/TutorialPanel';
import type { NodeInstance } from '../../core/graph';

type Tab = 'nodes' | 'inspector';

interface SavedBlock {
  filename: string;
  name: string;
  description: string;
  preset?: boolean;
}

interface Props {
  // Palette
  savedBlocks: SavedBlock[];
  onDeleteBlock: (filename: string) => void;
  // Inspector
  node: NodeInstance | null;
  selectedCount: number;
  onPropertyChange: (nodeId: string, key: string, value: unknown) => void;
  onSaveBlock: (nodeId: string) => void;
  graphJson: string;
}

export function LeftRail({
  savedBlocks,
  onDeleteBlock,
  node,
  selectedCount,
  onPropertyChange,
  onSaveBlock,
  graphJson,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [tab, setTab] = useState<Tab>('nodes');

  // Auto-switch tab on selection change: a node selected → Inspector,
  // selection cleared → Nodes. Manual tab clicks override until the next change.
  // Adjusting state during render (guarded by a changed value) is React's
  // recommended pattern for deriving state from props without an effect.
  const [prevNodeId, setPrevNodeId] = useState<string | null>(null);
  const nodeId = node?.id ?? null;
  if (nodeId !== prevNodeId) {
    setPrevNodeId(nodeId);
    setTab(nodeId ? 'inspector' : 'nodes');
  }

  // The rail defaults to open on the Nodes tab, so the palette is visible from
  // the start — emit the tutorial event once so the "Open the node palette"
  // task reflects reality. (Also fires on Nodes-tab click via showNodes.)
  useEffect(() => {
    if (!collapsed && tab === 'nodes') tutorialEvent('palette-opened');
    // run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function showNodes() {
    setTab('nodes');
    setCollapsed(false);
    tutorialEvent('palette-opened');
  }

  // Tab key toggles the rail (preserves the old palette shortcut).
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Tab') {
        e.preventDefault();
        setCollapsed((c) => {
          if (c && tab === 'nodes') tutorialEvent('palette-opened');
          return !c;
        });
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tab]);

  if (collapsed) {
    return (
      <button
        className="left-rail-handle"
        onClick={() => { setCollapsed(false); if (tab === 'nodes') tutorialEvent('palette-opened'); }}
        title="Expand panel (Tab)"
      >
        <Blocks size={15} />
        <ChevronRight size={16} />
      </button>
    );
  }

  return (
    <div className="left-rail">
      <div className="left-rail-tabs">
        <button
          className={`left-rail-tab ${tab === 'nodes' ? 'left-rail-tab-active' : ''}`}
          onClick={showNodes}
        >
          <Blocks size={14} />
          Nodes
        </button>
        <button
          className={`left-rail-tab ${tab === 'inspector' ? 'left-rail-tab-active' : ''}`}
          onClick={() => setTab('inspector')}
        >
          <SlidersHorizontal size={14} />
          Inspector
        </button>
        <button
          className="left-rail-collapse"
          onClick={() => setCollapsed(true)}
          title="Collapse panel (Tab)"
        >
          <ChevronLeft size={16} />
        </button>
      </div>

      <div className="left-rail-body">
        {tab === 'nodes' ? (
          <NodePalette savedBlocks={savedBlocks} onDeleteBlock={onDeleteBlock} />
        ) : (
          <PropertyInspector
            node={node}
            selectedCount={selectedCount}
            onPropertyChange={onPropertyChange}
            onSaveBlock={onSaveBlock}
            graphJson={graphJson}
          />
        )}
      </div>
    </div>
  );
}
