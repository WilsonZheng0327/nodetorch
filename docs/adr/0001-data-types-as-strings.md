# ADR-0001: Data types are strings, not enums

- **Status:** Accepted
- **Date:** 2026-07-05 *(backfilled from CLAUDE.md "Key Design Decisions")*

## Context

Ports carry typed data (tensors, scalars, images, token sequences, …), and
connection validation must decide whether an output port may feed an input port.
The obvious modelling choice is a closed enum of data types baked into the core.
But the core graph engine (Layers 1–4) is deliberately ML-agnostic — it must not
know that "tensor" or "loss" exist — while the ML domain (Layer 5) needs to add
new types freely without editing core code.

## Decision

Data types are plain **strings**, registered at runtime via a `DataTypeRegistry`
rather than declared as a compile-time enum. Compatibility between two type
strings is a registry lookup, not a hardcoded `switch`. The domain layer registers
every ML type it needs; the core never enumerates them.

## Consequences

- **Easier:** adding a data type is a `register()` call in the domain layer — no
  core edits, no enum to extend, no recompile of the engine's type surface. Keeps
  the ML/agnostic boundary clean (see [ADR-0003](0003-six-layer-ml-agnostic-architecture.md)).
- **Harder / cost:** no exhaustiveness checking from the compiler — a typo in a
  type string is a runtime mismatch, not a build error. Mitigated by centralizing
  registrations and testing `isValidConnection`.
- Type identity is nominal-by-string, so two conceptually different types must not
  reuse the same string.

## Alternatives considered

A TypeScript `enum` / union type: gives compile-time safety but forces every new
ML type through a core edit, breaking the layering rule. Rejected for that reason.
