// One place for the "POST JSON to the backend and normalize the outcome" pattern
// that was copy-pasted across useGraph's fetch calls. Collapses the three failure
// modes into a single tagged result so callers stop re-implementing the
// fetch → catch → json → check-status chain (and the "Cannot connect" string).

import { apiUrl } from './base';

export type BackendResult<T = unknown> = { ok: true; data: T } | { ok: false; error: string };

/**
 * POST `body` as JSON to a backend endpoint. Returns `{ ok: true, data }` with the
 * parsed body on success (callers read `.result` / `.results` off `data`), or
 * `{ ok: false, error }` for an unreachable backend, a non-JSON/HTTP error, or a
 * `{ status: 'error' }` payload. The error string is raw — wrap it with
 * `friendlyError` at the call site if it's shown to the user.
 */
export async function callBackend<T = unknown>(
  endpoint: string,
  body: unknown,
): Promise<BackendResult<T>> {
  let response: Response;
  try {
    response = await fetch(apiUrl(endpoint), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    return { ok: false, error: 'Cannot connect to backend — is the server running?' };
  }

  let data: { status?: string; error?: string } & Record<string, unknown>;
  try {
    data = await response.json();
  } catch {
    return { ok: false, error: `Backend error (HTTP ${response.status})` };
  }

  if (data.status !== 'ok') {
    return { ok: false, error: data.error ?? 'Request failed' };
  }
  return { ok: true, data: data as T };
}
