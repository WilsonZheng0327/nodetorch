# NodeTorch — Improvement Action Plan

Sequenced so each phase de-risks the next: CI first so every later change has a
regression net, bug fixes before refactors so broken code isn't relocated,
tooling before hygiene sweeps so the tools find the work, and type-tightening
before file splits so new module boundaries have real contracts.

**Rough total: ~2 weeks part-time, front-loaded** — after Phase 1 (~1.5 days in)
the repo is already strictly better-protected than today.

---

## Phase 0 — Lock in the green state (~2h)

Everything currently passes (`tsc -b`, 232 vitest tests, pytest suite), so CI is
pure upside and protects all later phases.

- [ ] Add `.github/workflows/ci.yml` with two jobs:
  - **frontend**: `npm ci` → `tsc -b` → `eslint` → `vitest run`
  - **backend**: `pip install -r requirements.txt` → `pytest`
- [ ] Use the CPU torch wheel index
      (`--index-url https://download.pytorch.org/whl/cpu`) — otherwise the
      backend job spends 10+ min pulling CUDA wheels
- [ ] Cache `~/.npm` and pip; target < 5 min total wall time
- [ ] ESLint warning **ratchet**: can't gate on `--max-warnings=0` yet
      (71 warnings), so record the current count and fail CI if it increases.
      Lower the budget as Phase 3 burns warnings down.

---

## Phase 1 — Fix the 7 ESLint errors (~half day)

These are real bugs, not style: `set-state-in-effect` causes double renders and
stale intermediate frames, and they'd be copied into any refactor of these files.

- [ ] `TrainingDashboard.tsx:94,100,118,125` — the `focusTestSignal` pattern
      (parent bumps a counter, child reacts in an effect) is an event modeled as
      state. Replace with a callback the parent invokes directly, or derive the
      tab from props during render.
- [ ] `LayerDetail.tsx:149`, `TokenizerDetail.tsx:48` — same class; if the
      intent is "reset local state when X changes," use a `key={x}` remount
      instead of an effect.
- [ ] `App.tsx:74` — React Compiler memoization bailout; usually a mutation or
      unstable closure in the component. Restructure until the compiler accepts
      it — the whole file loses auto-memoization otherwise.

---

## Phase 2 — Backend tooling + hygiene sweep (~1 day)

The TS side has four quality tools, Python has zero — add the tools first, then
let them drive the sweep.

- [ ] Add **ruff** (lint + format) and **mypy** via a single `pyproject.toml`;
      fold `pytest.ini` into it. Start mypy permissive
      (`ignore_missing_imports`, no strict flags) and ratchet per-module,
      strictest on `engine/` and `training/`.
- [ ] **Exception audit** — 89 `except Exception` blocks. Rule: every handler
      logs with context or re-raises; no bare `pass`. Kill the two silent
      swallows in training loops first (`diffusion.py:331`, `gan.py:402`) —
      they hide failures exactly where debugging is most expensive.
- [ ] **Fix private-name leaks** — `_safe_float`, `_last_run`, `_model_store`
      are imported across 10+ modules. Rename to public (`safe_float`) or wrap
      store access in a small `ModelStore` class. The underscore contract is
      already fiction; make the API honest.
- [ ] `topological_sort` in `build.py`: `queue.pop(0)` →
      `collections.deque` + `popleft()`. Two-line fix.
- [ ] `downloadBlob` in `useGraph.ts` catches *all* picker errors as
      "user cancelled" — a real write failure disappears. Narrow to
      `AbortError`; surface the rest.

---

## Phase 3 — Type recovery at the seams (~2–3 days)

Do this **before** splitting files: refactoring untyped
`Record<string, any>` contracts is guesswork; typed contracts make Phase 4
mechanical.

- [ ] Parameterize the core: `ExecutionContext<TProps, TInputs>` and
      `NodeDefinition<TProps>` with defaults of `Record<string, unknown>` — the
      engine stays generic, but each node file (e.g. `conv2d.ts`) declares
      `{ outChannels: number; kernelSize: number; ... }` and gets checked
      property access in its executors.
- [ ] Swap `any` → `unknown` in `src/core` (17 occurrences across `engine.ts`,
      `graph.ts`, `nodedef.ts`) and promote `no-explicit-any` to an **error**
      for `src/core` only — small enough to hold the line immediately.
- [ ] Enforce the cross-language contract currently maintained by comment:
      validate the WebSocket `EpochMessage` payload with a **zod schema** on the
      frontend mirroring the backend `TypedDict`. One boundary, two languages,
      one source of drift eliminated.
- [ ] Burn down the remaining ~54 `any` warnings opportunistically; the Phase 0
      ratchet prevents backsliding.

---

## Phase 4 — Decompose the god files (~3–4 days)

Now safe: CI green, effects fixed, contracts typed, and `useGraph.test.ts`
already exists as a harness.

- [ ] `useGraph.ts` (1,332 lines, 72 hook calls) → four hooks composed by a
      thin facade:
  - `useGraphState` — mutations + React Flow conversion
  - `useGraphIO` — serialize / download / load
  - `useTrainingSession` — WebSocket lifecycle + `EpochData`
  - `useSubgraphNav` — breadcrumbs

  Keep the facade's return shape identical so no consumer changes in the same PR.
- [ ] `export/model.py:_build_forward_body` (315 lines) and
      `graph_builder/detail.py:get_layer_detail` (248 lines) → per-node-type
      **dispatch tables**. The exact idiom already exists in `NODE_BUILDERS`;
      applying it here is consistency, not invention.
- [ ] `backprop_sim.py` (1,075 lines) → split into three modules:
      gradient-magnitude simulation, backward step-through, insight generation.

---

## Phase 5 — Strategic (as time allows)

- [ ] **One end-to-end smoke test**: through the real API, train an MNIST MLP
      for 1 epoch on CPU and assert loss decreased. Covers the seam none of the
      36 test files can — frontend-serialized graph → builder → training loop →
      WS messages — and it's the test that catches cross-language drift.
- [ ] **Decide the concurrency story** for `_model_store` / `_last_run`: two
      tabs currently share mutable globals. Either key the stores by session id
      or document single-session as a hard constraint next to the existing
      single-user note.
- [ ] Once warnings hit 0, flip CI to `--max-warnings=0` and delete the ratchet.

---

## Milestone summary

| Phase | Effort | Outcome |
|-------|--------|---------|
| 0 | ~2h | CI net over `tsc`, eslint (ratcheted), vitest, pytest |
| 1 | ~0.5d | 0 ESLint errors; render-cascade bugs fixed |
| 2 | ~1d | ruff + mypy in place; no silent exception swallows; public API honest |
| 3 | 2–3d | Typed executor contracts; `src/core` any-free; WS payload validated |
| 4 | 3–4d | No file > ~500 lines; dispatch tables replace 250–315-line functions |
| 5 | ongoing | E2E smoke test; concurrency decision; warnings → 0 |
