// Toolbar — save/load graph, run forward pass, train.

import { useRef, useState } from 'react';
import './Toolbar.css';

interface Props {
  onSave: () => string;
  onLoad: (json: string) => void;
  onRun: () => Promise<void>;
  onInfer: () => Promise<void>;
  onTrain: () => Promise<void>;
  onCancel: () => void;
  onClear: () => void;
  onOrganize: () => void;
  onShowAllViz: () => void;
  onHideAllViz: () => void;
  onStepThrough: () => void;
  onSimulateBackprop: () => void;
  status: { type: 'idle' | 'running' | 'success' | 'error'; message?: string };
  modelTrained: boolean;
  modelStale: boolean;
}

export function Toolbar({ onSave, onLoad, onRun, onInfer, onTrain, onCancel, onClear, onOrganize, onShowAllViz, onHideAllViz, onStepThrough, onSimulateBackprop, status, modelTrained, modelStale }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  async function handleSave() {
    const json = onSave();
    const blob = new Blob([json], { type: 'application/json' });

    // Use File System Access API if available (lets user pick name + location)
    if ('showSaveFilePicker' in window) {
      try {
        const handle = await (window as any).showSaveFilePicker({
          suggestedName: 'nodetorch-graph.json',
          types: [{
            description: 'NodeTorch Graph',
            accept: { 'application/json': ['.json'] },
          }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
        return;
      } catch {
        // User cancelled the picker — do nothing
        return;
      }
    }

    // Fallback: download to default location
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'nodetorch-graph.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleLoad() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        onLoad(reader.result);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  async function handleAction(fn: () => Promise<void>) {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="toolbar">
      <div className="toolbar-buttons">
        <button
          className="toolbar-btn toolbar-btn-run"
          onClick={() => handleAction(onRun)}
          disabled={busy}
          title="Run forward pass (random weights)"
        >
          Run
        </button>
        <button
          className={`toolbar-btn toolbar-btn-infer ${!modelTrained ? 'toolbar-btn-disabled-hint' : ''} ${modelStale ? 'toolbar-btn-stale' : ''}`}
          onClick={() => handleAction(onInfer)}
          disabled={busy}
          title={!modelTrained ? 'Train a model first' : modelStale ? 'Model outdated — retrain' : 'Infer using trained weights'}
        >
          Infer{modelStale ? ' !' : ''}
        </button>
        {busy ? (
          <button
            className="toolbar-btn toolbar-btn-cancel"
            onClick={onCancel}
            title="Cancel training"
          >
            Cancel
          </button>
        ) : (
          <button
            className="toolbar-btn toolbar-btn-train"
            onClick={() => handleAction(onTrain)}
            title="Train the model"
          >
            Train
          </button>
        )}
        <div className="toolbar-separator" />
        <button className="toolbar-btn" onClick={handleSave} title="Save graph to file">
          Save
        </button>
        <button className="toolbar-btn" onClick={handleLoad} title="Load graph from file">
          Load
        </button>
        <button className="toolbar-btn" onClick={onClear} disabled={busy} title="Clear all nodes and edges">
          Clear
        </button>
        <button className="toolbar-btn" onClick={onOrganize} disabled={busy} title="Auto-organize node layout">
          Organize
        </button>
        <button className="toolbar-btn" onClick={onShowAllViz} title="Show all viz panels">
          Show Viz
        </button>
        <button className="toolbar-btn" onClick={onHideAllViz} title="Hide all viz panels">
          Hide Viz
        </button>
        <button className="toolbar-btn" onClick={onStepThrough} title="Step through forward pass">
          Step Through
        </button>
        <button className="toolbar-btn" onClick={onSimulateBackprop} title="Animate one backward pass through the graph">
          Backprop
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </div>
      {status.message && status.type !== 'idle' && (
        <div className={`toolbar-status toolbar-status-${status.type}`}>
          {status.message.split('\n').map((line, i) => (
            <div key={i}>{line}</div>
          ))}
        </div>
      )}
    </div>
  );
}
