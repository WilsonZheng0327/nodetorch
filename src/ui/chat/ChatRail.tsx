// ChatRail — the integrated AI assistant panel. The agent loop runs in the
// backend; this renders the conversation, streams replies, and executes the
// agent's graph-edit tool calls locally via useGraph (see graphTools).

import './ChatRail.css';

import { useCallback, useContext, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { Sparkles, ChevronRight, ChevronLeft, ArrowUp, Square, Settings, Wrench } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { DomainCtx } from '../EngineNode';
import { buildNodeCatalog } from '../../domain/catalog';
import { useAgentChat } from './useAgentChat';
import { executeGraphTool, type GraphToolApi } from './graphTools';
import { AgentSettings } from './AgentSettings';

interface Props {
  /** Returns the current serialized graph JSON string (e.g. graph.saveGraph). */
  getGraphJson: () => string;
  /** Graph action surface for executing the agent's edits. */
  graph: GraphToolApi & {
    beginBatch: () => void;
    endBatch: () => void;
    savedBlocks: { filename: string; name: string; description: string }[];
  };
}

// Resizable width, persisted across reloads. Clamped to a sensible range.
const MIN_WIDTH = 320;
const MAX_WIDTH = 720;
const DEFAULT_WIDTH = 400;

function loadWidth(): number {
  const stored = Number(localStorage.getItem('chat-rail-width'));
  if (!stored || Number.isNaN(stored)) return DEFAULT_WIDTH;
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, stored));
}

export function ChatRail({ getGraphJson, graph }: Props) {
  const domain = useContext(DomainCtx);
  const [collapsed, setCollapsed] = useState(true);
  const [input, setInput] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [width, setWidth] = useState(loadWidth);

  // Drag the left edge to resize. Panel is anchored right, so dragging left
  // (smaller clientX) widens it: newWidth = startWidth + (startX - clientX).
  // Uses pointer capture so the drag keeps tracking even past the thin handle.
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  const onResizeDown = useCallback((e: ReactPointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = { startX: e.clientX, startWidth: width };
  }, [width]);

  const onResizeMove = useCallback((e: ReactPointerEvent) => {
    if (!dragRef.current) return;
    const { startX, startWidth } = dragRef.current;
    setWidth(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, startWidth + (startX - e.clientX))));
  }, []);

  const onResizeUp = useCallback((e: ReactPointerEvent) => {
    dragRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('chat-rail-width', String(width));
  }, [width]);

  // "E" toggles the assistant on/off (ignored while typing in a field).
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if ((e.target as HTMLElement)?.isContentEditable) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'e' || e.key === 'E') {
        e.preventDefault();
        setCollapsed((c) => !c);
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const catalog = useMemo(() => (domain ? buildNodeCatalog(domain) : []), [domain]);
  const getGraph = useMemo(
    () => () => {
      try {
        return JSON.parse(getGraphJson());
      } catch {
        return null;
      }
    },
    [getGraphJson],
  );

  const executeTool = useCallback(
    (name: string, args: Record<string, unknown>) =>
      domain ? executeGraphTool(graph, domain, name, args) : Promise.resolve('error: assistant not ready'),
    [graph, domain],
  );

  const { messages, status, error, send, cancel } = useAgentChat({
    getGraph,
    catalog,
    blocks: graph.savedBlocks,
    executeTool,
    beginBatch: graph.beginBatch,
    endBatch: graph.endBatch,
  });

  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages, status]);

  const streaming = status === 'streaming';

  function submit() {
    if (streaming || !input.trim()) return;
    send(input);
    setInput('');
  }

  if (collapsed) {
    return (
      <button className="chat-rail-handle" onClick={() => setCollapsed(false)} title="Open AI assistant (E)">
        <ChevronLeft size={16} />
        <Sparkles size={15} />
      </button>
    );
  }

  return (
    <div className="chat-rail" style={{ width }}>
      <div
        className="chat-rail-resize"
        onPointerDown={onResizeDown}
        onPointerMove={onResizeMove}
        onPointerUp={onResizeUp}
        title="Drag to resize"
      >
        <span className="chat-rail-resize-grip" />
      </div>
      <div className="chat-rail-header">
        <span className="chat-rail-title">
          <Sparkles size={15} />
          AI Assistant
        </span>
        <div className="chat-rail-header-actions">
          <button className="chat-rail-icon-btn" onClick={() => setSettingsOpen(true)} title="Provider settings">
            <Settings size={15} />
          </button>
          <button className="chat-rail-collapse" onClick={() => setCollapsed(true)} title="Collapse assistant (E)">
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="chat-rail-messages" ref={listRef}>
        {messages.length === 0 ? (
          <div className="chat-rail-empty">
            <Sparkles size={22} />
            <p className="chat-rail-empty-title">Ask me to build or explain</p>
            <p className="chat-rail-empty-desc">
              I can read your graph, explain it, and edit it — add or remove
              nodes, wire them up, and change properties. Pick a provider in
              settings (a paid API key or a local model), then ask away.
            </p>
            <button className="chat-rail-empty-btn" onClick={() => setSettingsOpen(true)}>
              <Settings size={13} /> Configure provider
            </button>
          </div>
        ) : (
          messages.map((m, i) =>
            m.role === 'tool' ? (
              <div key={i} className={`chat-rail-tool ${m.error ? 'chat-rail-tool-error' : ''}`}>
                <Wrench size={12} />
                {m.content}
              </div>
            ) : (
              <div key={i} className={`chat-rail-msg chat-rail-msg-${m.role}`}>
                {m.role === 'assistant' ? (
                  <div className="chat-rail-md">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  m.content
                )}
              </div>
            ),
          )
        )}
        {streaming && (
          <div className="chat-rail-working">
            <span className="chat-rail-typing">●●●</span> working…
          </div>
        )}
      </div>

      {error && <div className="chat-rail-error">{error}</div>}

      <div className="chat-rail-input-row">
        <textarea
          className="chat-rail-input"
          placeholder="Ask me to build or change your model…"
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        {streaming ? (
          <button className="chat-rail-send chat-rail-stop" onClick={cancel} title="Stop">
            <Square size={14} />
          </button>
        ) : (
          <button className="chat-rail-send" onClick={submit} disabled={!input.trim()} title="Send">
            <ArrowUp size={16} />
          </button>
        )}
      </div>

      <AgentSettings open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
