/** @vitest-environment jsdom */
// Hook-level tests for useAgentChat's tool bridge — in particular the approval
// gate: gated tools (start/stop training) pause for the user's Allow/Deny, and
// a deny returns a "denied:" observation without executing anything.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAgentChat } from '../../../src/ui/chat/useAgentChat';
import { setAskBeforeTraining } from '../../../src/ui/chat/agentPrefs';

// Minimal in-memory WebSocket (same shape as useGraph.test.ts) so we can drive
// the /agent message handler by hand.
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static OPEN = 1;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 1;
  sent: string[] = [];
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

function makeOpts() {
  return {
    getGraph: () => ({ version: '1.0', graph: { nodes: [], edges: [] } }),
    catalog: [],
    blocks: [],
    executeTool: vi.fn(async () => 'ok: done'),
    beginBatch: vi.fn(),
    endBatch: vi.fn(),
  };
}

// Node ≥22 ships its own (file-backed, here broken) localStorage global that
// shadows jsdom's — stub a deterministic in-memory one for agentPrefs.
function fakeStorage(): Storage {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => void m.set(k, String(v)),
    removeItem: (k: string) => void m.delete(k),
    clear: () => m.clear(),
    key: (i: number) => [...m.keys()][i] ?? null,
    get length() {
      return m.size;
    },
  } as Storage;
}

/** Render the hook, send one message, and open the socket. */
async function startTurn(opts: ReturnType<typeof makeOpts>) {
  const rendered = renderHook(() => useAgentChat(opts));
  let sendP: Promise<void> | undefined;
  act(() => {
    sendP = rendered.result.current.send('hi');
  });
  const ws = MockWebSocket.instances.at(-1)!;
  await act(async () => {
    ws.onopen?.();
    await sendP;
  });
  return { rendered, ws };
}

function toolResults(ws: MockWebSocket): { id: string; result: string }[] {
  return ws.sent.map((s) => JSON.parse(s)).filter((m) => m.type === 'tool_result');
}

describe('useAgentChat — tool bridge & approval gate', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal('localStorage', fakeStorage()); // fresh store: default pref = ask
    MockWebSocket.instances = [];
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('executes ungated tools immediately and replies with tool_result', async () => {
    const opts = makeOpts();
    const { rendered, ws } = await startTurn(opts);

    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({ type: 'tool_call', id: 't1', name: 'get_graph', args: {} }),
      });
    });
    expect(rendered.result.current.pendingApproval).toBeNull();
    expect(opts.executeTool).toHaveBeenCalledWith('get_graph', {});
    expect(toolResults(ws)).toEqual([{ type: 'tool_result', id: 't1', result: 'ok: done' }]);
  });

  it('pauses a gated tool for approval, then executes on Allow', async () => {
    const opts = makeOpts();
    const { rendered, ws } = await startTurn(opts);

    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({ type: 'tool_call', id: 't1', name: 'start_training', args: {} }),
      });
    });
    // Paused: card is up, nothing executed, no result sent yet.
    expect(rendered.result.current.pendingApproval?.name).toBe('start_training');
    expect(opts.executeTool).not.toHaveBeenCalled();
    expect(toolResults(ws)).toEqual([]);

    await act(async () => {
      rendered.result.current.resolveApproval(true);
    });
    expect(rendered.result.current.pendingApproval).toBeNull();
    expect(opts.executeTool).toHaveBeenCalledWith('start_training', {});
    expect(toolResults(ws).at(-1)?.result).toBe('ok: done');
  });

  it('Deny skips execution and returns a "denied:" observation', async () => {
    const opts = makeOpts();
    const { rendered, ws } = await startTurn(opts);

    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({ type: 'tool_call', id: 't2', name: 'stop_training', args: {} }),
      });
    });
    await act(async () => {
      rendered.result.current.resolveApproval(false);
    });
    expect(opts.executeTool).not.toHaveBeenCalled();
    expect(toolResults(ws).at(-1)?.result).toMatch(/^denied: the user declined stop_training/);
    // The visible tool line is flagged as denied.
    const toolMsg = rendered.result.current.messages.find((m) => m.role === 'tool');
    expect(toolMsg?.content).toBe('stop training — denied');
    expect(toolMsg?.error).toBe(true);
  });

  it('skips the gate entirely when the pref is off', async () => {
    setAskBeforeTraining(false);
    const opts = makeOpts();
    const { rendered, ws } = await startTurn(opts);

    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({ type: 'tool_call', id: 't3', name: 'start_training', args: {} }),
      });
    });
    expect(rendered.result.current.pendingApproval).toBeNull();
    expect(opts.executeTool).toHaveBeenCalledWith('start_training', {});
    expect(toolResults(ws).at(-1)?.result).toBe('ok: done');
  });

  it('cancel denies a pending approval before sending the WS cancel', async () => {
    const opts = makeOpts();
    const { rendered, ws } = await startTurn(opts);

    await act(async () => {
      ws.onmessage?.({
        data: JSON.stringify({ type: 'tool_call', id: 't4', name: 'start_training', args: {} }),
      });
    });
    expect(rendered.result.current.pendingApproval).not.toBeNull();

    await act(async () => {
      rendered.result.current.cancel();
    });
    expect(rendered.result.current.pendingApproval).toBeNull();
    expect(opts.executeTool).not.toHaveBeenCalled();
    // Denied tool_result goes out (backend ignores it for a cancelled turn),
    // followed by the cancel itself.
    const types = ws.sent.map((s) => JSON.parse(s).type);
    expect(types).toContain('cancel');
    expect(toolResults(ws).at(-1)?.result).toMatch(/^denied/);
  });
});
