# Frontend Architecture

How the NodeTorch frontend is layered, *why* it's layered that way, and a
walkthrough of what happens when you edit a node.

The frontend is a 6-layer stack. Each layer imports **only from the layers below
it** — never sideways into a sibling's internals, never upward. The bottom four
layers (`src/core/`) are a generic node-graph engine that **knows nothing about
machine learning**; all ML knowledge is quarantined in Layer 5 (`src/domain/`),
which fills the generic engine with ML content at startup.

```
Layer 6: UI            → src/ui/          React + React Flow; the useGraph hook
Layer 5: ML Domain     → src/domain/      ALL ML knowledge — every node definition
Layer 4: Node Registry → src/core/nodedef.ts   what a node TYPE is; the catalog
Layer 3: Execution     → src/core/engine.ts     walks the graph, runs executors
Layer 2: Type System   → src/core/datatypes.ts  port data-type compatibility
Layer 1: Graph Core    → src/core/graph.ts        nodes, edges, topology — pure data
```

## The one rule, and why it pays off

**A layer may import only from layers below it.** The import graph proves it
holds:

```
graph.ts      (L1)  imports: nothing                    ← true foundation
datatypes.ts  (L2)  imports: nothing
engine.ts     (L3)  imports: ./graph                    (L3 → L1)
nodedef.ts    (L4)  imports: ./engine (the Executor type)   (L4 → L3)
serialization.ts (L1) imports: ./graph                  (stays at the bottom)
```

"Lower" does **not** mean "more important." It means **depends on less**, and
therefore **more reusable and more testable in isolation**. `graph.ts` imports
nothing, so it's the bedrock. You can build, traverse, topologically sort, and
serialize a graph without any other layer present — and the core test suite does
exactly that.

This rule is enforced by a single recurring trick: **lower layers define generic
mechanisms and empty registries; higher layers inject the concrete content.**
You'll see it three times (the type registry, the mode registry, the node
registry), plus a dependency-injection variant (`ExecutorLookup`, `isKnownType`).

## Layer 1 — Graph Core (`src/core/graph.ts`)

The pure data structure. Imports nothing.

- `Graph` = `{ id, name, nodes: Map<string, NodeInstance>, edges: Edge[] }`.
- `NodeInstance` (`graph.ts:12`) = one node on the canvas:
  `{ id, type, position, properties, state, dirty, lastResult?, subgraph?, errors }`.
  The `type` is an **opaque string** (e.g. `"ml.layers.conv2d"`) — Layer 1 stores
  it but never resolves it. This is the **instance**, distinct from the
  **definition** of what Conv2d means (that's Layer 4).
- `Edge` (`graph.ts:57`) wires `source {nodeId, portId} → target {nodeId, portId}`.

Three responsibilities, nothing more:

1. **Topology** — `topologicalSort(graph)` gives nodes in dependency order.
2. **Dirty tracking** — `markDirty` flags a node and propagates downstream, so the
   engine can recompute only what changed.
3. **Structural mutation** — `createNode`/`addNode`/`addEdge`. These **throw** on
   bad references (`graph.ts:114` "Source node not found", `:127` "Edge already
   exists"), which is why untrusted input must be checked first (see Layer 1's
   serialization below).

Because the graph is type-blind, it **cannot** validate that a node's `type` is a
real ML node — `createNode` (`graph.ts:226`) stores any string verbatim. That's a
deliberate trade: purity at the bottom, validation at the boundaries.

### Serialization (`src/core/serialization.ts`, also Layer 1)

`SerializedGraph` (version `'1.0'`) is the plain-JSON projection of a `Graph`,
with nested `subgraph` support. It is the format used for **saving/loading files,
undo/redo snapshots, the clipboard, the saved-block library, and the payload sent
to the backend** — the single seam between the two halves of the app.

- `serializeGraph` / `deserializeGraph` — round-trip a `Graph` ↔ JSON.
- `validateSerializedGraph(data, isKnownType?)` — a pure pre-flight check. Run it
  before `deserializeGraph` so a malformed, outdated, or unknown-node file fails
  with a readable message instead of throwing half-way through `createNode`.

`validateSerializedGraph` takes an **`isKnownType` predicate** rather than
importing the Layer-4 registry. Checking "is this a real node type?" needs the
catalog, which lives four layers up — importing it would be an illegal upward
dependency (and a potential import cycle). Instead the validator declares an
abstract need, `(type) => boolean`, and the Layer-6 caller injects the
registry-backed implementation downward (`useGraph.ts`:
`(t) => domain.nodeRegistry.get(t) !== undefined`). The predicate is optional, so
the structural checks (version, duplicate ids, dangling edges) run with zero
dependencies — which is how the tests exercise them.

## Layer 2 — Type System (`src/core/datatypes.ts`)

A `DataTypeRegistry` that answers one question: **can this output port connect to
that input port?** (`isCompatible`, `datatypes.ts:49`). It also stores each type's
wire color.

It ships **empty** — it does not know `"tensor"` exists. Layer 5 registers the
actual types (`tensor`, `scalar`, `dataset`) at startup, with `scalar` declaring
`compatibleWith: ['tensor']` (a scalar is a 0-d tensor, so a scalar output may
feed a tensor input without an explicit converter). This registry is what powers
connection validation when you drag an edge on the canvas.

## Layer 3 — Execution Engine (`src/core/engine.ts`)

The generic graph-walker. Imports only Layer 1.

`ExecutionEngine.execute(graph, modeId, lookupExecutor)` (`engine.ts:97`):

1. Topologically sort the graph.
2. For each node: **skip if clean** (when `mode.caching` is on) → **gather inputs**
   by reading each upstream node's `lastResult.outputs` along the incoming edges
   (mapping `source.portId → target.portId`, so multi-output nodes work) → **run
   the node's executor** → **store the result** on `node.lastResult` and clear
   `dirty`.

The engine **doesn't know what any mode does**. An `ExecutionModeDefinition`
(`engine.ts:13`) is just `{ id, label, propagation, caching, executorKey }`;
"shape", "forward", and "train" are rows in a registry the engine walks
identically. It defines the contracts — `Executor`, `ExecutionContext` — and lets
higher layers implement them.

Two inversions to notice:

- **The engine doesn't import the registry.** It receives an `ExecutorLookup`
  function (`engine.ts:64`) — the same injection pattern as `isKnownType`. It
  declares "give me a way to find a node's executor"; Layer 6 passes
  `domain.nodeRegistry.getExecutor`.
- **The engine defines `Executor`** (`engine.ts:38`) and Layer 4 *imports that
  interface* to type its definitions. Lower layer owns the contract; higher layer
  conforms to it.

**Subgraph recursion** (`engine.ts:127-183`) is how composite blocks work: inject
the block's inputs into its inner `subgraph.input` sentinel, recursively
`execute` the inner graph, read the `subgraph.output` sentinel back out, and sum
the inner nodes' param counts onto the block.

## Layer 4 — Node Registry (`src/core/nodedef.ts`)

Defines what a node **type** *is*. Imports only Layer 3's `Executor` type.

- `NodeDefinition` (`nodedef.ts:85`): `type`, `displayName`, `category`,
  `getProperties()`, `getPorts(properties)`, and `executors: { shape, forward,
  train }`. Plus optional `validate`, `serialize`/`deserialize`, lifecycle hooks.
- `getPorts` is a **function of properties** (`nodedef.ts:113`), not a static
  list — that's what enables dynamic ports (e.g. Concat with N inputs).
- A node only implements the modes it supports. An optimizer has no `shape`
  executor, so `getExecutor` returns `undefined` and the engine skips it during
  shape inference.
- `NodeRegistry` is a `Map<type, NodeDefinition>` with `register` / `get` /
  `list(category?)` / `getExecutor(type, key)`.

Still ML-agnostic: it's a generic plugin catalog. It contains no actual nodes
until Layer 5 fills it.

## Layer 5 — ML Domain (`src/domain/`)

Where **all** ML knowledge lives, and where the three empty registries get
populated. `initDomain()` (`domain/index.ts:41`) is the one-time bootstrap:

```
const typeRegistry = new DataTypeRegistry();
const engine       = new ExecutionEngine();
const nodeRegistry = new NodeRegistry();

registerMLTypes(typeRegistry);   // tensor / scalar / dataset      → Layer 2
registerMLModes(engine);         // shape / forward / train        → Layer 3
for (const node of allNodes) nodeRegistry.register(node);   //     → Layer 4
```

- `ml-types.ts` adds the data types.
- `ml-modes.ts` adds the execution modes — `shape` is `eager` + caching (auto-runs
  on edits), `forward` is `manual` + caching, `train` is `manual` + no caching
  (every training step is a fresh pass).
- `nodes/<category>/` — each file is one `NodeDefinition`. Its `shape` executor is
  plain TypeScript math (e.g. Conv2d's `floor((H + 2P - K)/S) + 1`). That math,
  running in the browser, **is** the headline "instant shape inference" feature.

`initDomain()` returns a `DomainContext` — `{ typeRegistry, engine, nodeRegistry }`
— which Layer 6 holds and threads everywhere.

## Layer 6 — UI (`src/ui/`)

React + React Flow. The only layer allowed to touch the registry, the engine, and
the network at once.

- `useGraph.ts` is the bridge between the engine and React Flow (the main state
  hook). It owns the live `Graph`, runs `engine.execute(graph, 'shape', …)` after
  every edit, syncs results into React Flow nodes, and owns undo/redo,
  serialization, copy/paste, and **all** backend communication.
- `EngineNode.tsx` renders any node generically by reading its `NodeDefinition`
  (for ports/properties) and `lastResult.metadata` (for shapes, param counts,
  errors) — so new node types appear in the UI with no bespoke rendering code.

## Worked example: typing `outChannels = 64` on a Conv2d

1. **L6** `useGraph.updateProperty` writes the property, then calls **L1**
   `setProperty` → `markDirty`, flagging that node and everything downstream dirty.
2. **L6** calls **L3** `engine.execute(graph, 'shape', domain.nodeRegistry.getExecutor)`.
3. **L3** topo-sorts, skips clean nodes, reaches Conv2d, gathers its input shape
   from the upstream node's `lastResult`, and calls Conv2d's **L5** `shape` executor.
4. **L5** computes `[B, 64, H', W']` and returns it as `ExecutionResult.metadata`.
5. **L3** stores it on `node.lastResult`, marks the node clean, and continues to
   the next dirty node downstream (which recomputes against the new shape).
6. **L6** reads `lastResult.metadata` and re-renders the node.

The entire loop is **L6 → L1 → L3 → L5** and back, all in the browser, in
microseconds — no backend involved. This is why shape feedback is instant.

## Where the backend comes in

Shape inference is the only mode that runs entirely in the frontend. The heavy
modes — **train, infer, test, step-through, export** — need real PyTorch, so the
frontend serializes the graph (`serializeGraph`) and sends it to the backend over
WebSocket (training) or REST (everything else). The backend has its **own** engine
(`backend/engine/graph_builder/`) that does the *same topological walk* but builds
real `nn.Module`s and pushes tensors through them. The two halves are connected
only by the `SerializedGraph` JSON contract — which is why
`validateSerializedGraph` guards that seam. See `docs/backend-architecture.md` for
the other side of the wire.

## Key design decisions

- **Data types are strings, not enums**, registered via `DataTypeRegistry`. New
  types need no core changes.
- **`getPorts()` is a function of properties** — enables dynamic ports.
- **Execution modes are registered, not hardcoded** — adding a mode doesn't touch
  engine code.
- **`metadata` on `ExecutionResult` is unstructured** — each node emits whatever
  its visualization needs; the engine never interprets it.
- **Lower layers define interfaces and stay pure; higher layers inject
  concretions** (`ExecutorLookup`, `isKnownType`, the three registries). Nothing
  ever imports upward.
- **The graph is type-blind by design** — purity at the bottom, with validation at
  the boundaries (`validateSerializedGraph` on load, `validateForward` /
  `validateTraining` before running).

## Related docs

- `docs/backend-architecture.md` — the other side of the wire (graph → PyTorch).
- `docs/shape-inference.md` — how the shape executors compute dimensions.
- `docs/training-flow.md` — the frontend half of the train flow.
- `docs/custom-blocks.md` — subgraph blocks (Layer 3 recursion in depth).
- `docs/undo-redo.md`, `docs/copy-paste.md` — the `useGraph` subsystems.
