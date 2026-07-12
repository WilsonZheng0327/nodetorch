# NodeTorch Documentation

Start with [`CLAUDE.md`](../CLAUDE.md) at the repo root — it's the architecture
map and the fastest way to orient. This folder holds the deep dives.

Docs are organized by **purpose** (loosely following [Diátaxis](https://diataxis.fr)):
*explanation* answers "why / how does this work," *how-to* answers "help me do X,"
*reference* is lookup facts, and *decisions* record the "why" behind choices that
shaped the codebase.

## Explanation — how the system works & why

The 6-layer design and the two execution walkthroughs are the best entry points.

- [`explanation/frontend-architecture.md`](explanation/frontend-architecture.md) — the 6-layer frontend stack, why it's layered, and what happens when you edit a node.
- [`explanation/backend-architecture.md`](explanation/backend-architecture.md) — graph → PyTorch translation, with a line-by-line "what happens when you press Train."
- [`explanation/shape-inference.md`](explanation/shape-inference.md) — the "shape" execution mode: pure-TypeScript dimension math, edit to displayed shape.
- [`explanation/training-flow-walkthrough.md`](explanation/training-flow-walkthrough.md) — function-level call trace of the WebSocket Train flow (a `/code-explain` output), with subsystem drill-in points.
- [`explanation/training-plugins.md`](explanation/training-plugins.md) — the training-paradigm plugin system (standard / GAN / diffusion / autoregressive).
- [`explanation/visualization.md`](explanation/visualization.md) — extracting, transmitting, and rendering weight/activation stats.
- [`explanation/agent.md`](explanation/agent.md) — the provider-agnostic AI assistant (backend agent loop + browser tool-bridge).

## Feature deep dives

Cross-cutting features that touch many layers:

- [`features/custom-datasets.md`](features/custom-datasets.md) — the `data.custom` node: a spec-driven dataset interpreter + node-aware resolvers that fit any HuggingFace dataset.
- [`features/custom-blocks.md`](features/custom-blocks.md) — subgraphs: grouping nodes into one reusable block.
- [`features/multi-output-nodes.md`](features/multi-output-nodes.md) — how LSTM/GRU tuple outputs are handled end-to-end.
- [`features/undo-redo.md`](features/undo-redo.md) — snapshot-based undo/redo.
- [`features/copy-paste.md`](features/copy-paste.md) — copying and pasting selections of nodes.

## Reference — lookup facts

- [`reference/core-abstractions.md`](reference/core-abstractions.md) — flat glossary of the load-bearing types and registries (frontend `src/core/` + backend), each with a one-line purpose and where it lives.

## How-to

- **Adding a new node type** — see the step-by-step in [`CLAUDE.md`](../CLAUDE.md#adding-a-new-node-type).

## Decisions

- [`adr/`](adr/) — Architecture Decision Records: dated, immutable notes on *why*
  a significant choice was made, and what it costs. See [`adr/README.md`](adr/README.md).
  The condensed list lives in [`CLAUDE.md`](../CLAUDE.md#key-design-decisions).

## Bug log

Short postmortems of notable bugs — kept when the root cause exposed a wrong
assumption worth remembering. See [`bugs/`](bugs/README.md).

- [`bugs/seqpool-last-reads-padding.md`](bugs/seqpool-last-reads-padding.md) — `sequence_pool` `"last"` pooled over padding on short text, silently pinning a classifier at 50%.

## Surveys

Point-in-time codebase audits (dated snapshots, not living docs):

- [`surveys/survey-2026-07-05.md`](surveys/survey-2026-07-05.md)
- [`surveys/survey-2026-06-21.md`](surveys/survey-2026-06-21.md)

---

### Adding a doc

- **Explaining how something works or why?** → an explanation doc here.
- **A significant, hard-to-reverse choice with trade-offs?** → an [ADR](adr/).
- **Steps to accomplish a task?** → a how-to (inline in `CLAUDE.md` if short).
- **Facts about one file/interface?** → prefer a JSDoc/docstring next to the code,
  where it can't drift out of sync.

Keep this index in sync when you add a file.
