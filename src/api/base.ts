// The backend boundary — the single place that knows where the NodeTorch backend
// lives. Mirrors `backend/api/` on the server side. The UI calls through these
// helpers instead of hardcoding `http://localhost:8000`, so the base URL is
// configured in ONE spot and can point at a remote / GPU backend.
//
// Override with the VITE_API_BASE env var at build/dev time, e.g.
//   VITE_API_BASE=https://my-gpu-box:8000 npm run dev

const RAW_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://localhost:8000';

/** Base HTTP URL of the backend (no trailing slash). */
export const API_BASE = RAW_BASE.replace(/\/$/, '');

/** Base WebSocket URL of the backend (http→ws, https→wss). */
export const WS_BASE = API_BASE.replace(/^http/, 'ws');

/** Absolute REST URL for a path, e.g. apiUrl('/infer'). */
export function apiUrl(path: string): string {
  return API_BASE + path;
}

/** Absolute WebSocket URL for a path, e.g. wsUrl('/ws'). */
export function wsUrl(path: string): string {
  return WS_BASE + path;
}
