// useAgentChat — drives the /agent WebSocket: one streamed reply per turn.
// The agent loop lives in the backend; this hook just sends {message, graph,
// catalog} and assembles the streamed text_delta events into a message list.

import { useCallback, useEffect, useRef, useState } from 'react';
import type { NodeCatalog } from '../../domain/catalog';

const AGENT_WS_URL = 'ws://localhost:8000/agent';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export type ChatStatus = 'idle' | 'streaming' | 'error';

interface Options {
  /** Returns the current serialized graph object to send with each turn. */
  getGraph: () => unknown;
  /** The node catalog (sent once per connection). */
  catalog: NodeCatalog;
}

function newSessionId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `s-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}

export function useAgentChat({ getGraph, catalog }: Options) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<ChatStatus>('idle');
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionIdRef = useRef<string>(newSessionId());
  // Catalog is resent once per (re)connection so a backend restart re-seeds it.
  const catalogSentRef = useRef(false);

  // Append streamed text to the in-flight (last) assistant message.
  const appendToAssistant = useCallback((text: string) => {
    setMessages((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.role !== 'assistant') return prev;
      return [...prev.slice(0, -1), { ...last, content: last.content + text }];
    });
  }, []);

  const handleMessage = useCallback(
    (ev: MessageEvent) => {
      let msg: { type?: string; text?: string; error?: string };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case 'text_delta':
          appendToAssistant(msg.text ?? '');
          break;
        case 'done':
        case 'cancelled':
          setStatus('idle');
          break;
        case 'error':
          setError(msg.error || 'Agent error');
          setStatus('error');
          // Drop the empty assistant placeholder if nothing streamed.
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === 'assistant' && last.content === '') return prev.slice(0, -1);
            return prev;
          });
          break;
      }
    },
    [appendToAssistant],
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
      ws.onerror = () => {
        reject(new Error('Could not reach the agent. Is the backend running?'));
      };
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
      setMessages((prev) => [
        ...prev,
        { role: 'user', content: trimmed },
        { role: 'assistant', content: '' },
      ]);
      setStatus('streaming');
      try {
        const ws = await ensureSocket();
        const payload: Record<string, unknown> = {
          type: 'chat',
          message: trimmed,
          sessionId: sessionIdRef.current,
          graph: getGraph(),
        };
        if (!catalogSentRef.current) {
          payload.catalog = catalog;
          catalogSentRef.current = true;
        }
        ws.send(JSON.stringify(payload));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus('error');
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === 'assistant' && last.content === '') return prev.slice(0, -1);
          return prev;
        });
      }
    },
    [status, ensureSocket, getGraph, catalog],
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
