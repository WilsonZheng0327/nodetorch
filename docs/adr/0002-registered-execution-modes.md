# ADR-0002: Execution modes are registered, not hardcoded

- **Status:** Accepted
- **Date:** 2026-07-05 *(backfilled from CLAUDE.md "Key Design Decisions")*

## Context

NodeTorch runs a graph in several ways: "shape" (eager TypeScript dimension math,
no backend), "train" (WebSocket streaming), "test" (held-out evaluation), and
"infer" (trained weights in backend memory). More paradigms are plausible over
time. If the execution engine hardcodes a fixed set of modes with a `switch`,
every new mode means editing engine code — the engine ends up knowing about ML
concerns it was designed to stay out of.

## Decision

Execution modes are **registered** against the engine via an
`ExecutionModeDefinition` + `Executor` contract (`src/core/engine.ts`), the same
way node definitions and data types are registered. The engine knows how to *run
an executor*; it does not enumerate which modes exist. The ML domain registers
"shape", "train", "test", "infer" at bootstrap (`initDomain()`).

## Consequences

- **Easier:** adding a mode is a registration in the domain layer — the engine is
  untouched. Modes stay co-located with the ML knowledge they need.
- **Easier:** keeps Layer 3 (engine) ML-agnostic, consistent with
  [ADR-0003](0003-six-layer-ml-agnostic-architecture.md) and
  [ADR-0001](0001-data-types-as-strings.md).
- **Harder / cost:** the set of available modes is only known at runtime after
  registration, so tooling can't statically enumerate them; discovery is via the
  registry.

## Alternatives considered

A hardcoded `ExecutionMode` enum + `switch` in the engine: simpler to read, but
couples the engine to the mode list and violates the layering rule. Rejected.
