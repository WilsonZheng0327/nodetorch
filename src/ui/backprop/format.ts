// Number formatting for backprop views. The implementation now lives in the
// shared `src/ui/format.ts`; this file re-exports it so existing imports keep
// working. Prefer importing from `src/ui/format` in new code.
export { fmt, signed } from '../format';
