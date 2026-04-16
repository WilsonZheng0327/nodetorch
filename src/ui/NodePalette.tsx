// Node palette — collapsible sidebar listing all available node types.
// Organized by nested categories. Drag a node type onto the canvas to add it.

import './NodePalette.css';

import { useContext, useState, useEffect, type DragEvent } from 'react';
import type { NodeDefinition } from '../core/nodedef';
import { DomainCtx } from './EngineNode';

// Tree structure for nested categories
interface CategoryNode {
  name: string;
  items: NodeDefinition[];
  children: Map<string, CategoryNode>;
}

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

  return root;
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
        <span className="palette-folder-icon">{expanded ? '▾' : '▸'}</span>
        {node.name}
      </button>

      {expanded && (
        <div className="palette-folder-content">
          {/* Items at this level */}
          {node.items.map((def) => (
            <div
              key={def.type}
              className="palette-item"
              draggable
              onDragStart={(e) => onDragStart(e, def.type)}
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
  const [collapsed, setCollapsed] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'Tab') {
        e.preventDefault();
        setCollapsed((c) => !c);
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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
    <div className="palette">
      <button
        className="palette-toggle"
        onClick={() => setCollapsed(!collapsed)}
        title={collapsed ? 'Expand palette (Tab)' : 'Collapse palette (Tab)'}
      >
        Nodes {collapsed ? '+' : '-'}
      </button>

      {!collapsed && (
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
              <div className="palette-folder palette-folder-root">
                Saved Blocks
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
      )}
    </div>
  );
}
