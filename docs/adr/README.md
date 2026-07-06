# Architecture Decision Records

An **ADR** captures a single significant decision: the context that forced it, the
choice made, and the consequences. One decision per file, numbered sequentially.

## Rules

- **Immutable once accepted.** Don't rewrite history. If a decision changes,
  write a new ADR that *supersedes* the old one and update the old one's status
  with a link. This is the whole point: an ADR shows what you knew *at the time*.
- **One decision per file**, `NNNN-short-kebab-title.md`.
- **Short.** Roughly a page. If it's growing, the explanation belongs in a
  `docs/*.md` deep dive; link to it.
- Record it when a choice is **significant and costly to reverse** — an API shape,
  a layering rule, a build-time contract. Skip the trivial and the obvious.

## Status values

`Proposed` → `Accepted` → (later) `Superseded by [ADR-NNNN](...)` or `Deprecated`.

## Index

| # | Title | Status |
|---|-------|--------|
| [0001](0001-data-types-as-strings.md) | Data types are strings, not enums | Accepted |
| [0002](0002-registered-execution-modes.md) | Execution modes are registered, not hardcoded | Accepted |
| [0003](0003-six-layer-ml-agnostic-architecture.md) | Six-layer architecture; layers 1–4 know nothing about ML | Accepted |

## New ADR

Copy [`template.md`](template.md), take the next number, add a row above.
