// ChatRail — the integrated AI assistant panel. The agent loop runs in the
// backend; this renders the conversation and streams replies via useAgentChat.
// v1 is explain-only: it answers questions about the current graph.

import './ChatRail.css';

import { useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Sparkles, ChevronRight, ChevronLeft, ArrowUp, Square, Settings } from 'lucide-react';
import { DomainCtx } from '../EngineNode';
import { buildNodeCatalog } from '../../domain/catalog';
import { useAgentChat } from './useAgentChat';
import { AgentSettings } from './AgentSettings';

interface Props {
  /** Returns the current serialized graph JSON string (e.g. graph.saveGraph). */
  getGraphJson: () => string;
}

export function ChatRail({ getGraphJson }: Props) {
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

  const { messages, status, error, send, cancel } = useAgentChat({ getGraph, catalog });

  const listRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages]);

  const streaming = status === 'streaming';

  function submit() {
    if (streaming || !input.trim()) return;
    send(input);
    setInput('');
  }

  if (collapsed) {
    return (
      <button
        className="chat-rail-handle"
        onClick={() => setCollapsed(false)}
        title="Open AI assistant"
      >
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
          <button
            className="chat-rail-icon-btn"
            onClick={() => setSettingsOpen(true)}
            title="Provider settings"
          >
            <Settings size={15} />
          </button>
          <button
            className="chat-rail-collapse"
            onClick={() => setCollapsed(true)}
            title="Collapse assistant"
          >
            <ChevronRight size={16} />
          </button>
        </div>
      </div>

      <div className="chat-rail-messages" ref={listRef}>
        {messages.length === 0 ? (
          <div className="chat-rail-empty">
            <Sparkles size={22} />
            <p className="chat-rail-empty-title">Ask about your model</p>
            <p className="chat-rail-empty-desc">
              I can read your graph and explain what it does, why it might not
              train, or what a node means. Pick a provider in settings (a paid
              API key or a local model), then ask away.
            </p>
            <button className="chat-rail-empty-btn" onClick={() => setSettingsOpen(true)}>
              <Settings size={13} /> Configure provider
            </button>
          </div>
        ) : (
          messages.map((m, i) => {
            const isStreamingPlaceholder =
              streaming && i === messages.length - 1 && m.role === 'assistant' && m.content === '';
            return (
              <div key={i} className={`chat-rail-msg chat-rail-msg-${m.role}`}>
                {isStreamingPlaceholder ? <span className="chat-rail-typing">…</span> : m.content}
              </div>
            );
          })
        )}
      </div>

      {error && <div className="chat-rail-error">{error}</div>}

      <div className="chat-rail-input-row">
        <textarea
          className="chat-rail-input"
          placeholder="Ask about your model…"
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
          <button
            className="chat-rail-send"
            onClick={submit}
            disabled={!input.trim()}
            title="Send"
          >
            <ArrowUp size={16} />
          </button>
        )}
      </div>

      <AgentSettings open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </div>
  );
}
