// Polls the backend so the UI can show a clear "backend offline" banner. A
// student who forgot to start the server otherwise sees only scattered,
// transient errors when they try to train/infer.

import { useEffect, useState } from 'react';
import { apiUrl } from '../api/base';

/** Returns `false` while the backend is unreachable. Checks on mount and every
 *  `intervalMs`. Starts optimistic (`true`) to avoid a banner flash on load. */
export function useBackendHealth(intervalMs = 10000): boolean {
  const [online, setOnline] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(apiUrl('/system-info'), { signal: AbortSignal.timeout(4000) });
        if (!cancelled) setOnline(res.ok);
      } catch {
        if (!cancelled) setOnline(false);
      }
    };
    check();
    const timer = setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [intervalMs]);

  return online;
}
