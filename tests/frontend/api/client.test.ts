import { describe, it, expect, afterEach, vi } from 'vitest';
import { callBackend } from '../../../src/api/client';

const origFetch = global.fetch;
afterEach(() => {
  global.fetch = origFetch;
});

function mockFetch(impl: () => Promise<unknown> | never) {
  // @ts-expect-error test stub
  global.fetch = vi.fn(impl);
}

describe('callBackend', () => {
  it('returns ok + parsed data on a { status: "ok" } response', async () => {
    mockFetch(() => Promise.resolve({ json: () => Promise.resolve({ status: 'ok', result: 42 }) }));
    const r = await callBackend('/x', {});
    expect(r).toEqual({ ok: true, data: { status: 'ok', result: 42 } });
  });

  it('returns a connection error when fetch throws', async () => {
    mockFetch(() => Promise.reject(new Error('down')));
    const r = await callBackend('/x', {});
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/Cannot connect to backend/);
  });

  it('returns an HTTP error when the body is not JSON', async () => {
    mockFetch(() =>
      Promise.resolve({ status: 500, json: () => Promise.reject(new Error('not json')) }),
    );
    const r = await callBackend('/x', {});
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).toMatch(/HTTP 500/);
  });

  it('surfaces a { status: "error" } payload error verbatim', async () => {
    mockFetch(() =>
      Promise.resolve({ json: () => Promise.resolve({ status: 'error', error: 'boom' }) }),
    );
    const r = await callBackend('/x', {});
    expect(r).toEqual({ ok: false, error: 'boom' });
  });
});
