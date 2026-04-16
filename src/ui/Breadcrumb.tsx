// Breadcrumb navigation — shows the path from root into nested subgraphs.
// Click a breadcrumb to navigate back to that level.

import './Breadcrumb.css';

interface NavEntry {
  graphId: string;
  label: string;
  nodeId: string;
}

interface Props {
  navStack: NavEntry[];
  onNavigate: (depth: number) => void;
}

export function Breadcrumb({ navStack, onNavigate }: Props) {
  if (navStack.length === 0) return null;

  return (
    <div className="breadcrumb">
      <button className="breadcrumb-item" onClick={() => onNavigate(0)}>
        Root
      </button>
      {navStack.map((entry, i) => (
        <span key={entry.graphId}>
          <span className="breadcrumb-sep">/</span>
          <button
            className={`breadcrumb-item ${i === navStack.length - 1 ? 'breadcrumb-current' : ''}`}
            onClick={() => onNavigate(i + 1)}
          >
            {entry.label}
          </button>
        </span>
      ))}
    </div>
  );
}
