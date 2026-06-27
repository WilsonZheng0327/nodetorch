// Toolbar — save/load graph, step through, train/test/infer.

import { useRef, useState, useEffect } from 'react';
import { Download, Upload, BookOpen, Trash2, LayoutGrid, Eye, EyeOff, Footprints, Undo2, ChevronDown, GraduationCap, FileCode } from 'lucide-react';
import { tutorialEvent } from './tutorial/TutorialPanel';
import './Toolbar.css';
import { apiUrl } from '../api/base';

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
  onSaveModel: () => Promise<void>;            // bundle: graph + weights (.ntmodel)
  onLoadModel: (file: File) => Promise<void>;  // bundle: replaces graph + weights
  onSaveWeights: () => Promise<void>;          // weights only (.pt)
  onLoadWeights: (file: File) => Promise<void>; // weights onto the current graph
  onExportPython: () => Promise<void>;
  status: { type: 'idle' | 'running' | 'success' | 'error'; message?: string };
  modelTrained: boolean;
  modelStale: boolean;
}

export function Toolbar({ onSave, onLoad, onInfer, onTrain, onTest, onCancel, onClear, onOrganize, onShowAllViz, onHideAllViz, onStepThrough, onSimulateBackprop, onSaveModel, onLoadModel, onSaveWeights, onLoadWeights, onExportPython, status, modelTrained, modelStale }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);       // graph (.json)
  const modelInputRef = useRef<HTMLInputElement>(null);      // model bundle (.ntmodel)
  const weightsInputRef = useRef<HTMLInputElement>(null);    // weights (.pt)
  const [busy, setBusy] = useState(false);
  const [presets, setPresets] = useState<{filename: string; name: string}[]>([]);
  const [presetsOpen, setPresetsOpen] = useState(false);
  const presetsRef = useRef<HTMLDivElement>(null);
  // Save / Load are each a single button opening a 3-item dropdown (Graph / Model / Weights).
  const [saveOpen, setSaveOpen] = useState(false);
  const [loadOpen, setLoadOpen] = useState(false);
  const saveMenuRef = useRef<HTMLDivElement>(null);
  const loadMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!presetsOpen && !saveOpen && !loadOpen) return;
    const handler = (e: PointerEvent | MouseEvent) => {
      const t = e.target as Node;
      if (presetsRef.current && !presetsRef.current.contains(t)) setPresetsOpen(false);
      if (saveMenuRef.current && !saveMenuRef.current.contains(t)) setSaveOpen(false);
      if (loadMenuRef.current && !loadMenuRef.current.contains(t)) setLoadOpen(false);
    };
    document.addEventListener('pointerdown', handler, true);
    return () => document.removeEventListener('pointerdown', handler, true);
  }, [presetsOpen, saveOpen, loadOpen]);

  async function openPresets() {
    if (presetsOpen) { setPresetsOpen(false); return; }
    try {
      const res = await fetch(apiUrl('/presets'));
      const data = await res.json();
      if (data.status === 'ok') setPresets(data.presets);
    } catch { /* backend not running */ }
    setPresetsOpen(true);
  }

  async function loadPreset(filename: string) {
    try {
      const res = await fetch(apiUrl('/presets/load'), {
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

  // Loading a model bundle replaces both the graph and the weights — confirm first.
  function handleModelFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    if (!window.confirm('Loading a model will replace your current graph and weights. Continue?')) return;
    handleAction(() => onLoadModel(file));
  }

  // Loading weights only keeps the current graph (architecture must match) — no warning.
  function handleWeightsFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    handleAction(() => onLoadWeights(file));
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
        <div ref={saveMenuRef} style={{ position: 'relative' }}>
          <button className="toolbar-btn toolbar-btn-icon" onClick={() => { setSaveOpen((o) => !o); setLoadOpen(false); }} title="Save graph, model, or weights">
            <Download size={15} /><ChevronDown size={11} />
          </button>
          {saveOpen && (
            <div className="toolbar-presets-dropdown">
              <button className="toolbar-presets-item" onClick={() => { setSaveOpen(false); handleSave(); }}>Graph (.json)</button>
              <button className="toolbar-presets-item" disabled={!modelTrained} title={!modelTrained ? 'Train a model first' : undefined} onClick={() => { setSaveOpen(false); handleAction(onSaveModel); }}>Model (.ntmodel)</button>
              <button className="toolbar-presets-item" disabled={!modelTrained} title={!modelTrained ? 'Train a model first' : undefined} onClick={() => { setSaveOpen(false); handleAction(onSaveWeights); }}>Weights (.pt)</button>
            </div>
          )}
        </div>
        <div ref={loadMenuRef} style={{ position: 'relative' }}>
          <button className="toolbar-btn toolbar-btn-icon" onClick={() => { setLoadOpen((o) => !o); setSaveOpen(false); }} title="Load graph, model, or weights">
            <Upload size={15} /><ChevronDown size={11} />
          </button>
          {loadOpen && (
            <div className="toolbar-presets-dropdown">
              <button className="toolbar-presets-item" onClick={() => { setLoadOpen(false); handleLoad(); }}>Graph (.json)</button>
              <button className="toolbar-presets-item" onClick={() => { setLoadOpen(false); modelInputRef.current?.click(); }}>Model (.ntmodel)</button>
              <button className="toolbar-presets-item" onClick={() => { setLoadOpen(false); weightsInputRef.current?.click(); }}>Weights (.pt)</button>
            </div>
          )}
        </div>
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
        <button className="toolbar-btn toolbar-btn-icon" onClick={() => handleAction(onExportPython)} disabled={busy} title="Export as standalone Python file">
          <FileCode size={15} />
        </button>
        <div className="toolbar-separator" />
        <button className="toolbar-btn toolbar-btn-icon" onClick={() => window.dispatchEvent(new Event('nodetorch-tutorial-reopen'))} title="Open tutorial">
          <GraduationCap size={15} />
        </button>
        <input ref={fileInputRef} type="file" accept=".json" style={{ display: 'none' }} onChange={handleFileChange} />
        <input ref={modelInputRef} type="file" accept=".ntmodel" style={{ display: 'none' }} onChange={handleModelFile} />
        <input ref={weightsInputRef} type="file" accept=".pt" style={{ display: 'none' }} onChange={handleWeightsFile} />
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
