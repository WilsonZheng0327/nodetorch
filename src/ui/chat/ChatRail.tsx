// ChatRail — the integrated AI assistant panel. The agent loop runs in the
// backend; this renders the conversation, streams replies, and executes the
// agent's graph-edit tool calls locally via useGraph (see graphTools).

import './ChatRail.css';

import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Sparkles, ChevronRight, ChevronLeft, ArrowUp, Square, Settings, Wrench } from 'lucide-react';
import { DomainCtx } from '../EngineNode';
import { buildNodeCatalog } from '../../domain/catalog';
import { useAgentChat } from './useAgentChat';
import { executeGraphTool, type GraphToolApi } from './graphTools';
import { AgentSettings } from './AgentSettings';

interface Props {
  /** Returns the current serialized graph JSON string (e.g. graph.saveGraph). */
  getGraphJson: () => string;
  /** Graph action surface for executing the agent's edits. */
  graph: GraphToolApi & { beginBatch: () => void; endBatch: () => void };
}

export function ChatRail({ getGraphJson, graph }: Props) {
  const domain = useContext(DomainCtx);
  const [collapsed, setCollapsed] = useState(false);
  const [input, setInput] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);

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
      <button className="chat-rail-handle" onClick={() => setCollapsed(false)} title="Open AI assistant">
        <ChevronLeft size={16} />
        <Sparkles size={15} />
      </button>
    );
  }

  return (
    <div className="chat-rail">
      <div className="chat-rail-header">
        <span className="chat-rail-title">
          <Sparkles size={15} />
          AI Assistant
        </span>
        <div className="chat-rail-header-actions">
          <button className="chat-rail-icon-btn" onClick={() => setSettingsOpen(true)} title="Provider settings">
            <Settings size={15} />
          </button>
          <button className="chat-rail-collapse" onClick={() => setCollapsed(true)} title="Collapse assistant">
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
                {m.content}
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
