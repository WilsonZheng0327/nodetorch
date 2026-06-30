// Node palette — content-only list of all available node types.
// Organized by nested categories. Drag a node type onto the canvas to add it.
// Rendered inside LeftRail (which owns the panel chrome, collapse, and tabs).

import './NodePalette.css';

import { useContext, useState, type DragEvent } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { NodeDefinition } from '../core/nodedef';
import { DomainCtx } from './contexts';

// Tree structure for nested categories
interface CategoryNode {
  name: string;
  items: NodeDefinition[];
  children: Map<string, CategoryNode>;
}

// Explicit ordering for top-level categories. Anything not listed falls after
// these, alphabetically. Keeps "Custom Block" at the end regardless of name.
const TOP_LEVEL_ORDER = ['Data', 'ML', 'Custom Block'];

// One-line explanation shown under each top-level ("big") category header.
const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  Data: 'Datasets that feed the model — image and text sources to train and test on.',
  ML: 'The network itself — layers, activations, losses, and optimizers.',
  'Custom Block': 'Reusable sub-graphs you build once and drop into any model.',
};

function buildCategoryTree(defs: NodeDefinition[]): Map<string, CategoryNode> {
  const root = new Map<string, CategoryNode>();

  for (const def of defs) {
    const path = def.category;
    let level = root;

    for (let i = 0; i < path.length; i++) {
      const name = path[i];
      if (!level.has(name)) {
        level.set(name, { name, items: [], children: new Map() });
      }
      const node = level.get(name)!;

      // If this is the last category level, add the item here
      if (i === path.length - 1) {
        node.items.push(def);
      } else {
        level = node.children;
      }
    }
  }

  sortTree(root);
  return orderTopLevel(root);
}

// Re-order just the root level by TOP_LEVEL_ORDER (sortTree already sorted
// nested levels alphabetically).
function orderTopLevel(root: Map<string, CategoryNode>): Map<string, CategoryNode> {
  const rank = (name: string) => {
    const i = TOP_LEVEL_ORDER.indexOf(name);
    return i === -1 ? TOP_LEVEL_ORDER.length : i;
  };
  const entries = [...root.entries()].sort((a, b) => {
    const r = rank(a[0]) - rank(b[0]);
    return r !== 0 ? r : a[0].localeCompare(b[0]);
  });
  return new Map(entries);
}

function sortTree(level: Map<string, CategoryNode>) {
  // Sort items alphabetically within each category
  for (const node of level.values()) {
    node.items.sort((a, b) => a.displayName.localeCompare(b.displayName));
    if (node.children.size > 0) sortTree(node.children);
  }
  // Sort categories alphabetically
  const sorted = [...level.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  level.clear();
  for (const [k, v] of sorted) level.set(k, v);
}

function onDragStart(event: DragEvent, nodeType: string) {
  event.dataTransfer.setData('application/nodetorch-type', nodeType);
  event.dataTransfer.effectAllowed = 'move';
}

// Recursive category renderer
function CategoryGroup({ node, depth }: { node: CategoryNode; depth: number }) {
  const [expanded, setExpanded] = useState(depth === 0);
  const hasContent = node.items.length > 0 || node.children.size > 0;

  if (!hasContent) return null;

  return (
    <>
      {/* Category header — same indentation as sibling items */}
      <button
        className={`palette-folder ${depth === 0 ? 'palette-folder-root' : ''}`}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="palette-folder-icon">{expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}</span>
        {node.name}
      </button>

      {depth === 0 && CATEGORY_DESCRIPTIONS[node.name] && (
        <div className="palette-folder-desc">{CATEGORY_DESCRIPTIONS[node.name]}</div>
      )}

      {expanded && (
        <div className="palette-folder-content">
          {/* Items at this level */}
          {node.items.map((def) => (
            <div
              key={def.type}
              className="palette-item"
              draggable
              onDragStart={(e) => onDragStart(e, def.type)}
              title={def.description}
            >
              <span className="palette-item-name">{def.displayName}</span>
              <span className="palette-item-desc">{def.description}</span>
            </div>
          ))}

          {/* Subcategories */}
          {Array.from(node.children.values()).map((child) => (
            <CategoryGroup key={child.name} node={child} depth={depth + 1} />
          ))}
        </div>
      )}
    </>
  );
}

interface SavedBlock {
  filename: string;
  name: string;
  description: string;
  preset?: boolean;
}

interface PaletteProps {
  savedBlocks: SavedBlock[];
  onDeleteBlock: (filename: string) => void;
}

export function NodePalette({ savedBlocks, onDeleteBlock }: PaletteProps) {
  const domain = useContext(DomainCtx);
  const [search, setSearch] = useState('');

  if (!domain) return null;

  const allDefs = domain.nodeRegistry.list();
  const query = search.toLowerCase();
  const filteredDefs = query
    ? allDefs.filter((d) =>
        d.displayName.toLowerCase().includes(query) ||
        d.description.toLowerCase().includes(query) ||
        d.category.some((c) => c.toLowerCase().includes(query)),
      )
    : allDefs;
  const tree = buildCategoryTree(filteredDefs);

  return (
    <div className="palette-content">
      <input
        className="palette-search"
        type="text"
        placeholder="Search nodes..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      {Array.from(tree.values()).map((node) => (
        <CategoryGroup key={node.name} node={node} depth={0} />
      ))}

      {/* Saved blocks */}
      {savedBlocks.length > 0 && (!query || savedBlocks.some((b) => b.name.toLowerCase().includes(query))) && (
        <>
          <div className="palette-divider" />
          <div className="palette-folder palette-folder-root">
            Saved Blocks
          </div>
          <div className="palette-folder-desc">
            Custom blocks you've saved plus shipped presets — drag any onto the canvas.
          </div>
          <div className="palette-folder-content">
            {savedBlocks
              .filter((b) => !query || b.name.toLowerCase().includes(query))
              .map((block) => (
                <div key={block.filename} className="palette-item palette-saved-block">
                  <div
                    className="palette-saved-block-drag"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData('application/nodetorch-block', block.filename);
                      e.dataTransfer.effectAllowed = 'move';
                    }}
                  >
                    <span className="palette-item-name">{block.name}</span>
                    <span className="palette-item-desc">{block.description}</span>
                  </div>
                  {!block.preset && <button
                    className="palette-saved-block-delete"
                    onClick={() => onDeleteBlock(block.filename)}
                    title="Delete block"
                  >
                    &times;
                  </button>}
                </div>
              ))}
          </div>
        </>
      )}

      {filteredDefs.length === 0 && savedBlocks.length === 0 && (
        <div className="palette-empty">No matching nodes</div>
      )}
    </div>
  );
}
