# ADR-0003: Six-layer architecture; layers 1–4 know nothing about ML

- **Status:** Accepted
- **Date:** 2026-07-05 *(backfilled from CLAUDE.md "Key Design Decisions")*

## Context

NodeTorch is a node-based visual editor that happens to be about ML. Two kinds of
concern are easy to entangle: the generic machinery of a node graph (nodes, edges,
topological order, dirty tracking, a type system, an execution engine) and the ML
domain knowledge (what a Conv2d is, what a loss is, how shapes propagate). If ML
concepts leak into the graph core, the core becomes hard to reason about, hard to
test in isolation, and every ML change risks destabilizing the engine.

## Decision

The frontend is a strict **6-layer stack**, each layer depending only on layers
below it:

```
Layer 6: UI            → src/ui/        (React + React Flow)
Layer 5: ML Domain     → src/domain/    (register() calls — ALL ML knowledge)
Layer 4: Node Registry → src/core/nodedef.ts
Layer 3: Execution Engine → src/core/engine.ts
Layer 2: Type System   → src/core/datatypes.ts
Layer 1: Graph Core    → src/core/graph.ts
```

**Layers 1–4 know nothing about ML.** All ML knowledge — node types, data types,
execution modes, shape math — is registered from Layer 5. The backend mirrors the
engine: the graph builder does the same topological walk with real PyTorch tensors.

## Consequences

- **Easier:** the core is a reusable, testable node-graph engine; ML lives in one
  place. Adding a node type, data type ([ADR-0001](0001-data-types-as-strings.md)),
  or execution mode ([ADR-0002](0002-registered-execution-modes.md)) is a
  registration in Layer 5, never a core edit.
- **Easier:** the layering rule gives a clear answer to "where does this code go?"
  and makes accidental coupling reviewable — an import that points sideways or
  upward is a smell.
- **Harder / cost:** indirection. Behavior that a monolith would inline is split
  across a registry boundary, so tracing a feature means following registrations.
  Some genuinely ML-shaped generality (dynamic ports) has to be expressed in
  ML-agnostic terms in the core (`getPorts()` as a function of properties).
- Discipline required: the value evaporates the moment an ML string or concept is
  hardcoded below Layer 5. Guard it in review.

## Alternatives considered

A conventional feature-oriented structure with ML concepts throughout: less
indirection, faster to write initially, but the graph engine could never be
reasoned about or tested independent of ML, and every ML change would reach into
core. Rejected — the layered split is the project's central bet. See
[`frontend-architecture.md`](../explanation/frontend-architecture.md) for the full walkthrough.
