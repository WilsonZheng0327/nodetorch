// useAgentChat — drives the /agent WebSocket. Streams text, and bridges the
// agent's tool calls: a tool_call from the backend is executed locally (via the
// graph tools) and a tool_result is sent back. The whole turn is one undo step
// (beginBatch on the first tool call, endBatch when the turn ends).

import { useCallback, useEffect, useRef, useState } from 'react';
import type { NodeCatalog } from '../../domain/catalog';

const AGENT_WS_URL = 'ws://localhost:8000/agent';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string;
  error?: boolean; // for tool messages whose result was an error
}

export type ChatStatus = 'idle' | 'streaming' | 'error';

interface Options {
  /** Returns the current serialized graph object to send with each turn. */
  getGraph: () => unknown;
  /** The node catalog (sent once per connection). */
  catalog: NodeCatalog;
  /** Apply a tool call to the graph; returns an observation string. */
  executeTool: (name: string, args: Record<string, unknown>) => Promise<string>;
  /** Group this turn's edits into a single undo step. */
  beginBatch: () => void;
  endBatch: () => void;
}

function newSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `s-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}

function describeTool(name: string, args: Record<string, unknown>): string {
  switch (name) {
    case 'set_node_property':
      return `set ${args.nodeId}.${args.key} = ${JSON.stringify(args.value)}`;
    case 'add_node':
      return `add ${args.type}`;
    case 'connect':
      return `connect ${args.sourceId}.${args.sourcePort} → ${args.targetId}.${args.targetPort}`;
    case 'remove_node':
      return `remove ${args.nodeId}`;
    default:
      return name;
  }
}

export function useAgentChat(opts: Options) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  // Keep the latest callbacks reachable from the (stable) socket handler.
  const optsRef = useRef(opts);
  useEffect(() => {
    optsRef.current = opts;
  });

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>(newSessionId());
  const catalogSentRef = useRef(false);
  const batchOpenRef = useRef(false);

  const appendToAssistant = useCallback((text: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === 'assistant') {
        return [...prev.slice(0, -1), { ...last, content: last.content + text }];
      }
      return [...prev, { role: 'assistant', content: text }];
    });
  }, []);

  const endTurn = useCallback(() => {
    if (batchOpenRef.current) {
      optsRef.current.endBatch();
      batchOpenRef.current = false;
    }
  }, []);

  const handleMessage = useCallback(
    async (ev: MessageEvent) => {
      let msg: { type?: string; text?: string; error?: string; id?: string; name?: string; args?: Record<string, unknown> };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }

      switch (msg.type) {
        case 'text_delta':
          appendToAssistant(msg.text ?? '');
          break;

        case 'tool_call': {
          // First edit of the turn opens the one-undo batch.
          if (!batchOpenRef.current) {
            optsRef.current.beginBatch();
            batchOpenRef.current = true;
          }
          const name = msg.name ?? '';
          const args = msg.args ?? {};
          const result = await optsRef.current.executeTool(name, args);
          wsRef.current?.send(JSON.stringify({ type: 'tool_result', id: msg.id, result }));
          setMessages((prev) => [
            ...prev,
            { role: 'tool', content: describeTool(name, args), error: result.startsWith('error') },
          ]);
          break;
        }

        case 'done':
        case 'cancelled':
          endTurn();
          setStatus('idle');
          break;

        case 'error':
          endTurn();
          setError(msg.error || 'Agent error');
          setStatus('error');
          break;
      }
    },
    [appendToAssistant, endTurn],
  );

  const ensureSocket = useCallback((): Promise<WebSocket> => {
    const existing = wsRef.current;
    if (existing && existing.readyState === WebSocket.OPEN) return Promise.resolve(existing);

    return new Promise((resolve, reject) => {
      const ws = new WebSocket(AGENT_WS_URL);
      wsRef.current = ws;
      catalogSentRef.current = false; // fresh connection → resend catalog
      ws.onopen = () => resolve(ws);
      ws.onmessage = handleMessage;
      ws.onerror = () => reject(new Error('Could not reach the agent. Is the backend running?'));
      ws.onclose = () => {
        if (wsRef.current === ws) wsRef.current = null;
      };
    });
  }, [handleMessage]);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || status === 'streaming') return;
      setError(null);
      setMessages((prev) => [...prev, { role: 'user', content: trimmed }]);
      setStatus('streaming');
      try {
        const ws = await ensureSocket();
        const payload: Record<string, unknown> = {
          type: 'chat',
          message: trimmed,
          sessionId: sessionIdRef.current,
          graph: optsRef.current.getGraph(),
        };
        if (!catalogSentRef.current) {
          payload.catalog = optsRef.current.catalog;
          catalogSentRef.current = true;
        }
        ws.send(JSON.stringify(payload));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus('error');
      }
    },
    [status, ensureSocket],
  );

  const cancel = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ type: 'cancel' }));
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, []);

  return { messages, status, error, send, cancel };
}
