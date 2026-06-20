// ChatRail — docked right panel that will host the integrated AI agent.
//
// This is a UI scaffold only: the agent backend is not wired yet. The message
// list, input, and send affordance are in place so the layout is final, but
// sending is a no-op until the agent transport lands. See the TODO seam below.

import './ChatRail.css';

import { useState } from 'react';
import { Sparkles, ChevronRight, ChevronLeft, ArrowUp } from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export function ChatRail() {
  const [collapsed, setCollapsed] = useState(false);
  const [input, setInput] = useState('');
  const [messages] = useState<ChatMessage[]>([]);

  // TODO(agent): replace this no-op with the agent transport. The agent should
  // receive `input` plus the current serialized graph, stream a response, and
  // be able to act on the graph (add/connect/configure nodes, run modes).
  const canSend = false;
  function handleSend() {
    if (!canSend) return;
    // intentionally unimplemented — wired when the agent backend exists
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
        <button
          className="chat-rail-collapse"
          onClick={() => setCollapsed(true)}
          title="Collapse assistant"
        >
          <ChevronRight size={16} />
        </button>
      </div>

      <div className="chat-rail-messages">
        {messages.length === 0 ? (
          <div className="chat-rail-empty">
            <Sparkles size={22} />
            <p className="chat-rail-empty-title">Your AI assistant lives here</p>
            <p className="chat-rail-empty-desc">
              Soon you'll be able to ask it to build, explain, and debug your
              model right alongside the canvas.
            </p>
            <span className="chat-rail-badge">Coming soon</span>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`chat-rail-msg chat-rail-msg-${m.role}`}>
              {m.content}
            </div>
          ))
        )}
      </div>

      <div className="chat-rail-input-row">
        <textarea
          className="chat-rail-input"
          placeholder="Ask the assistant… (coming soon)"
          rows={1}
          value={input}
          disabled={!canSend}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <button
          className="chat-rail-send"
          onClick={handleSend}
          disabled={!canSend || !input.trim()}
          title="Send"
        >
          <ArrowUp size={16} />
        </button>
      </div>
    </div>
  );
}
