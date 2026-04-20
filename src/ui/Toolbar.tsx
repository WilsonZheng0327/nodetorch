// Toolbar — save/load graph, run forward pass, train.

import { useRef, useState, useEffect } from 'react';
import { Download, Upload, BookOpen, Trash2, LayoutGrid, Eye, EyeOff, Footprints, Undo2, HardDriveDownload, HardDriveUpload, GraduationCap } from 'lucide-react';
import { tutorialEvent } from './tutorial/TutorialPanel';
import './Toolbar.css';

interface Props {
  onSave: () => string;
  onLoad: (json: string) => void;
  onInfer: () => Promise<void>;
  onTrain: () => Promise<void>;
  onTest: () => Promise<void>;
  onCancel: () => void;
  onClear: () => void;
  onOrganize: () => void;
  onShowAllViz: () => void;
  onHideAllViz: () => void;
  onStepThrough: () => void;
  onSimulateBackprop: () => void;
  onSaveModel: () => Promise<void>;
  onLoadModel: () => Promise<void>;
  status: { type: 'idle' | 'running' | 'success' | 'error'; message?: string };
  modelTrained: boolean;
  modelStale: boolean;
}

export function Toolbar({ onSave, onLoad, onInfer, onTrain, onTest, onCancel, onClear, onOrganize, onShowAllViz, onHideAllViz, onStepThrough, onSimulateBackprop, onSaveModel, onLoadModel, status, modelTrained, modelStale }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [presets, setPresets] = useState<{filename: string; name: string}[]>([]);
  const [presetsOpen, setPresetsOpen] = useState(false);
  const presetsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!presetsOpen) return;
    const handler = (e: MouseEvent) => {
      if (presetsRef.current && !presetsRef.current.contains(e.target as Node)) {
        setPresetsOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [presetsOpen]);

  async function openPresets() {
    if (presetsOpen) { setPresetsOpen(false); return; }
    try {
      const res = await fetch('http://localhost:8000/presets');
      const data = await res.json();
      if (data.status === 'ok') setPresets(data.presets);
    } catch { /* backend not running */ }
    setPresetsOpen(true);
  }

  async function loadPreset(filename: string) {
    try {
      const res = await fetch('http://localhost:8000/presets/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      const data = await res.json();
      if (data.status === 'ok') {
        onLoad(JSON.stringify(data.data));
        setPresetsOpen(false);
        tutorialEvent('preset-loaded');
      }
    } catch { /* backend not running */ }
  }

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
        <button
          className={`toolbar-btn toolbar-btn-test ${!modelTrained ? 'toolbar-btn-disabled-hint' : ''}`}
          onClick={() => handleAction(onTest)}
          disabled={busy || !modelTrained}
          title={!modelTrained ? 'Train a model first' : 'Evaluate on the held-out test set'}
        >
          Test
        </button>
        <button
          className={`toolbar-btn toolbar-btn-infer ${!modelTrained ? 'toolbar-btn-disabled-hint' : ''} ${modelStale ? 'toolbar-btn-stale' : ''}`}
          onClick={() => handleAction(onInfer)}
          disabled={busy}
          title={!modelTrained ? 'Train a model first' : modelStale ? 'Model outdated — retrain' : 'Infer using trained weights'}
        >
          Infer{modelStale ? ' !' : ''}
        </button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn toolbar-btn-icon" onClick={handleSave} title="Save graph to file">
          <Download size={15} />
        </button>
        <button className="toolbar-btn toolbar-btn-icon" onClick={handleLoad} title="Load graph from file">
          <Upload size={15} />
        </button>
        <div ref={presetsRef} style={{ position: 'relative' }}>
          <button className="toolbar-btn toolbar-btn-icon" onClick={openPresets} title="Load a model preset">
            <BookOpen size={15} />
          </button>
          {presetsOpen && (
            <div className="toolbar-presets-dropdown">
              {presets.map((p) => (
                <button key={p.filename} className="toolbar-presets-item" onClick={() => loadPreset(p.filename)}>
                  {p.name}
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="toolbar-btn toolbar-btn-icon" onClick={onClear} disabled={busy} title="Clear all nodes and edges">
          <Trash2 size={15} />
        </button>
        <button className="toolbar-btn toolbar-btn-icon" onClick={onOrganize} disabled={busy} title="Auto-organize node layout">
          <LayoutGrid size={15} />
        </button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn toolbar-btn-icon" onClick={onShowAllViz} title="Show all viz panels">
          <Eye size={15} />
        </button>
        <button className="toolbar-btn toolbar-btn-icon" onClick={onHideAllViz} title="Hide all viz panels">
          <EyeOff size={15} />
        </button>
        <button className="toolbar-btn toolbar-btn-icon" onClick={onStepThrough} title="Step through forward pass">
          <Footprints size={15} />
        </button>
        <button className="toolbar-btn toolbar-btn-icon" onClick={onSimulateBackprop} title="Animate one backward pass through the graph">
          <Undo2 size={15} />
        </button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn toolbar-btn-icon" onClick={() => handleAction(onSaveModel)} disabled={busy || !modelTrained} title="Save trained weights to disk">
          <HardDriveDownload size={15} />
        </button>
        <button className="toolbar-btn toolbar-btn-icon" onClick={() => handleAction(onLoadModel)} disabled={busy} title="Load trained weights from disk">
          <HardDriveUpload size={15} />
        </button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn toolbar-btn-icon" onClick={() => window.dispatchEvent(new Event('nodetorch-tutorial-reopen'))} title="Open tutorial">
          <GraduationCap size={15} />
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
