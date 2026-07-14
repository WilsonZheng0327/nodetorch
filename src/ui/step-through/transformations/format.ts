// Number formatting for step-through transformation views. The implementation
// now lives in the shared `src/ui/format.ts`; this file re-exports it so existing
// imports keep working. Prefer importing from `src/ui/format` in new code.
export { fmtAxis, fmtValue } from '../../format';
