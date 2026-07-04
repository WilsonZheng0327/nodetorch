// LeftRail — docked left panel hosting the node palette and the property
// inspector behind two tabs. Selecting a node auto-switches to Inspector;
// clearing the selection falls back to Nodes. Either tab can be pinned
// manually. The Tab key collapses/expands the whole rail.

import './LeftRail.css';

import { useEffect, useState } from 'react';
import { Blocks, SlidersHorizontal, ChevronLeft, ChevronRight } from 'lucide-react';
import { NodePalette } from '../NodePalette';
import { PropertyInspector } from '../inspector/PropertyInspector';
import { tutorialEvent } from '../tutorial/tutorialEvent';
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
  // Per-click signals from App (bumped counters): open-to-inspector on node
  // click, collapse on empty-canvas click. Counters (not booleans) so repeat
  // clicks re-fire even when the selection is unchanged.
  inspectSignal: number;
  closeSignal: number;
  /** Reports open/closed state up so App can include the rail in Esc handling. */
  onOpenChange: (open: boolean) => void;
}

export function LeftRail({
  savedBlocks,
  onDeleteBlock,
  node,
  selectedCount,
  onPropertyChange,
  onSaveBlock,
  graphJson,
  inspectSignal,
  closeSignal,
  onOpenChange,
}: Props) {
  const [collapsed, setCollapsed] = useState(true);
  const [tab, setTab] = useState<Tab>('nodes');

  // Report open state up so App can include the rail in its Esc priority order.
  useEffect(() => {
    onOpenChange(!collapsed);
  }, [collapsed, onOpenChange]);

  // Auto-switch tab on selection change: a node selected → Inspector,
  // selection cleared → Nodes. Selecting a node also expands the rail if it's
  // collapsed, so its details are visible without manually opening the panel.
  // Manual tab clicks override until the next change. Adjusting state during
  // render (guarded by a changed value) is React's recommended pattern for
  // deriving state from props without an effect.
  const [prevNodeId, setPrevNodeId] = useState<string | null>(null);
  const nodeId = node?.id ?? null;
  if (nodeId !== prevNodeId) {
    setPrevNodeId(nodeId);
    setTab(nodeId ? 'inspector' : 'nodes');
    if (nodeId) setCollapsed(false);
  }

  // Explicit per-click signals from App. A node click opens the inspector even
  // when the selection is unchanged (re-clicking the same node); an empty-canvas
  // click collapses the rail entirely.
  const [prevInspect, setPrevInspect] = useState(inspectSignal);
  if (inspectSignal !== prevInspect) {
    setPrevInspect(inspectSignal);
    setTab('inspector');
    setCollapsed(false);
  }
  const [prevClose, setPrevClose] = useState(closeSignal);
  if (closeSignal !== prevClose) {
    setPrevClose(closeSignal);
    setCollapsed(true);
  }

  // The rail starts collapsed, so the palette isn't visible on mount. If
  // something opens it on the Nodes tab, emit the tutorial event so the
  // "Open the node palette" task reflects reality. (Also fires on Nodes-tab
  // click via showNodes and on expand.)
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

  // "1" toggles the rail; Tab (while open) switches between Nodes and Inspector.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === '1') {
        e.preventDefault();
        setCollapsed((c) => {
          if (c && tab === 'nodes') tutorialEvent('palette-opened');
          return !c;
        });
      } else if (e.key === 'Tab' && !collapsed) {
        e.preventDefault();
        const next = tab === 'nodes' ? 'inspector' : 'nodes';
        setTab(next);
        if (next === 'nodes') tutorialEvent('palette-opened');
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [tab, collapsed]);

  if (collapsed) {
    return (
      <button
        className="left-rail-handle"
        onClick={() => {
          setCollapsed(false);
          if (tab === 'nodes') tutorialEvent('palette-opened');
        }}
        title="Expand panel (1)"
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
          title="Collapse panel (1)"
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
