/** @vitest-environment jsdom */
// Hook-level tests for the load-bearing bits of useGraph: undo/redo + batching,
// and the connection validator. Uses renderHook with a STABLE domain (created
// once, like App's useMemo) so hook effects don't churn.

import { describe, it, expect, beforeAll } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { initDomain } from '../../../src/domain';
import { useGraph } from '../../../src/ui/useGraph';
import { validateTraining } from '../../../src/core/validation';

// jsdom has no backend; the mount effect (refreshBlocks) fetches /blocks. Stub
// fetch so it resolves to an empty list instead of hitting the network.
beforeAll(() => {
  // @ts-expect-error minimal fetch stub for the tests
  global.fetch = () =>
    Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok', blocks: [] }) });
});

describe('useGraph — undo/redo', () => {
  it('adds a node, undoes it, and redoes it', async () => {
    const domain = initDomain();
    const { result } = renderHook(() => useGraph(domain));
    expect(result.current.currentGraph.nodes.size).toBe(0);

    await act(async () => {
      await result.current.addNode('ml.activations.relu', { x: 0, y: 0 });
    });
    expect(result.current.currentGraph.nodes.size).toBe(1);

    await act(async () => {
      await result.current.undo();
    });
    expect(result.current.currentGraph.nodes.size).toBe(0);

    await act(async () => {
      await result.current.redo();
    });
    expect(result.current.currentGraph.nodes.size).toBe(1);
  });

  it('collapses a beginBatch/endBatch group into a single undo step', async () => {
    const domain = initDomain();
    const { result } = renderHook(() => useGraph(domain));

    await act(async () => {
      result.current.beginBatch();
      await result.current.addNode('ml.activations.relu', { x: 0, y: 0 });
      await result.current.addNode('ml.activations.sigmoid', { x: 100, y: 0 });
      result.current.endBatch();
    });
    expect(result.current.currentGraph.nodes.size).toBe(2);

    // One undo reverts the whole batch, not just the last node.
    await act(async () => {
      await result.current.undo();
    });
    expect(result.current.currentGraph.nodes.size).toBe(0);
  });
});

describe('useGraph — isValidConnection', () => {
  it('accepts a compatible tensor→tensor link and rejects the invalid cases', async () => {
    const domain = initDomain();
    const { result } = renderHook(() => useGraph(domain));

    let convId = '';
    let reluId = '';
    await act(async () => {
      convId = (await result.current.addNode('ml.layers.conv2d', { x: 0, y: 0 })) ?? '';
      reluId = (await result.current.addNode('ml.activations.relu', { x: 200, y: 0 })) ?? '';
    });

    const link = { source: convId, target: reluId, sourceHandle: 'out', targetHandle: 'in' };
    expect(result.current.isValidConnection(link)).toBe(true);

    // Rule 1: no self-connection.
    expect(
      result.current.isValidConnection({
        source: convId,
        target: convId,
        sourceHandle: 'out',
        targetHandle: 'in',
      }),
    ).toBe(false);

    // Missing node.
    expect(
      result.current.isValidConnection({
        source: convId,
        target: 'ghost',
        sourceHandle: 'out',
        targetHandle: 'in',
      }),
    ).toBe(false);

    // Rule 4: relu's single input port can't take a second edge.
    await act(async () => {
      result.current.connect(link);
    });
    expect(result.current.isValidConnection(link)).toBe(false);
  });
});

// Minimal in-memory WebSocket so we can drive runTrain's message handler.
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

describe('useGraph — training WebSocket handler', () => {
  it('appends each epoch message to trainingProgress as EpochData', async () => {
    const origWS = global.WebSocket;
    (global as unknown as { WebSocket: unknown }).WebSocket = MockWebSocket;
    MockWebSocket.instances = [];
    try {
      const domain = initDomain();
      const { result } = renderHook(() => useGraph(domain));

      // Build a minimal valid trainable graph: mnist → flatten → linear → loss,
      // mnist.labels → loss.labels, loss → adam.
      const ids: Record<string, string> = {};
      await act(async () => {
        ids.mnist = (await result.current.addNode('data.mnist', { x: 0, y: 0 })) ?? '';
        ids.flat = (await result.current.addNode('ml.layers.flatten', { x: 150, y: 0 })) ?? '';
        ids.lin = (await result.current.addNode('ml.layers.linear', { x: 300, y: 0 })) ?? '';
        ids.loss = (await result.current.addNode('ml.loss.cross_entropy', { x: 450, y: 0 })) ?? '';
        ids.opt = (await result.current.addNode('ml.optimizers.adam', { x: 600, y: 0 })) ?? '';
      });
      await act(async () => {
        result.current.connect({
          source: ids.mnist,
          target: ids.flat,
          sourceHandle: 'out',
          targetHandle: 'in',
        });
        result.current.connect({
          source: ids.flat,
          target: ids.lin,
          sourceHandle: 'out',
          targetHandle: 'in',
        });
        result.current.connect({
          source: ids.lin,
          target: ids.loss,
          sourceHandle: 'out',
          targetHandle: 'predictions',
        });
        result.current.connect({
          source: ids.mnist,
          target: ids.loss,
          sourceHandle: 'labels',
          targetHandle: 'labels',
        });
        result.current.connect({
          source: ids.loss,
          target: ids.opt,
          sourceHandle: 'out',
          targetHandle: 'loss',
        });
      });

      // Sanity: the graph is actually trainable (guards the fixture, not the SUT).
      expect(validateTraining(result.current.graph, domain.nodeRegistry)).toEqual([]);

      act(() => {
        result.current.runTrain();
      });
      const ws = MockWebSocket.instances.at(-1)!;
      expect(ws).toBeTruthy();
      act(() => ws.onopen?.());
      expect(ws.sent.length).toBe(1); // train message sent on open

      act(() =>
        ws.onmessage?.({
          data: JSON.stringify({
            type: 'epoch',
            epoch: 1,
            totalEpochs: 2,
            loss: 0.5,
            accuracy: 0.9,
          }),
        }),
      );
      expect(result.current.trainingProgress).toHaveLength(1);
      expect(result.current.trainingProgress[0].loss).toBe(0.5);
      expect(result.current.trainingProgress[0].accuracy).toBe(0.9);

      act(() =>
        ws.onmessage?.({
          data: JSON.stringify({
            type: 'epoch',
            epoch: 2,
            totalEpochs: 2,
            loss: 0.3,
            accuracy: 0.95,
          }),
        }),
      );
      expect(result.current.trainingProgress).toHaveLength(2);

      // Finish cleanly so the runTrain promise resolves.
      act(() =>
        ws.onmessage?.({
          data: JSON.stringify({
            type: 'train_result',
            status: 'ok',
            results: { nodeResults: {} },
          }),
        }),
      );
      expect(result.current.trainingActive).toBe(false);
    } finally {
      (global as unknown as { WebSocket: unknown }).WebSocket = origWS;
    }
  });
});
