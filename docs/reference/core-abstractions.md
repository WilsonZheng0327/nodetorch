# Core Abstractions Reference

A flat lookup of the load-bearing types and registries the system is built on —
"what is `Executor` vs `ExecutionContext` again, and where does it live?" For the
*narrative* of how they fit together, read
[`../explanation/frontend-architecture.md`](../explanation/frontend-architecture.md)
and [`../explanation/backend-architecture.md`](../explanation/backend-architecture.md);
for the *why* behind the big choices, see [`../adr/`](../adr/).

Descriptions are distilled from the JSDoc / docstrings on each symbol — those are
the source of truth. Symbols are cited by file + name (not line number) so this
doesn't drift; grep the name to jump to it.

---

## Frontend (`src/core/`, Layers 1–4 — ML-agnostic)

### Graph model — `graph.ts` (Layer 1)

| Symbol | What it is |
|--------|-----------|
| `Graph` | Top-level container: a map of nodes + a list of edges. A subgraph is itself a `Graph` nested on a `NodeInstance`. |
| `NodeInstance` | One node *on the canvas* (a specific Conv2d "conv1" at a position, with property values + runtime state). The *definition* of what Conv2d means lives in `NodeDefinition`. |
| `Edge` | A directed connection from a source port to a target port; data flows source → target. |
| `ExecutionResult` | What an executor returns for a node: `outputs` (portId → data), optional `state`, and unstructured `metadata` (shapes, param counts, viz) the UI reads. Layer 1 stores it, never interprets it. |
| `ValidationMessage` | An error/warning/info string attached to a node for display. |

### Type system — `datatypes.ts` (Layer 2)

| Symbol | What it is |
|--------|-----------|
| `DataTypeDefinition` | A data type that flows through ports (`id`, `label`, wire `color`, optional `compatibleWith` for implicit conversions). |
| `DataTypeRegistry` | Holds all registered data types and answers "can this output connect to that input?" Populated by Layer 5. → [ADR-0001](../adr/0001-data-types-as-strings.md) (types are strings, not enums). |

### Node definition — `nodedef.ts` (Layer 4)

| Symbol | What it is |
|--------|-----------|
| `NodeDefinition` | The complete definition of a node *type* — display info, `getProperties()`, `getPorts()`, its `executors` map, and optional `validate`/lifecycle/serialize hooks. Layer 5 creates and registers these. |
| `PortDefinition` | A connection point: `direction` (input/output), `dataType`, `allowMultiple`, `optional`. |
| `PropertyDefinition` | An editable property (e.g. `kernelSize`); drives the auto-generated inspector widget. `affects` says whether a change recomputes ports, re-runs executors, or both. |
| `PropertyType` | The kind of value a property holds (number / string / boolean / select / range / custom) — picks the UI widget. |
| `NodeRegistry` | Plugin-style store of all `NodeDefinition`s. UI reads `list()` for the palette; engine reads `getExecutor()`. |
| `ValidationResult` | A message returned by `NodeDefinition.validate()`. |

> `getPorts(properties)` is a **function**, not a static list — so ports can be
> computed from properties (e.g. Concat's N inputs). This is the dynamic-ports
> pattern noted in `CLAUDE.md`.

### Execution — `engine.ts` (Layer 3)

| Symbol | What it is |
|--------|-----------|
| `ExecutionEngine` | Walks nodes in topological order and calls each one's executor. Knows *when* and *in what order* to run — not *what* the executors do. |
| `ExecutionModeDefinition` | Declares how a mode behaves: `propagation` (eager vs manual), `caching`, and which `executorKey` to look up. Registered by Layer 5. → [ADR-0002](../adr/0002-registered-execution-modes.md). |
| `Executor` | The thing that actually runs on a node: `execute(context) → ExecutionResult`. Interface defined here, implemented in Layer 5. |
| `ExecutionContext` | Everything an executor needs: `inputs` (portId → upstream data), `properties`, `state`, and the current `mode`. |
| `ExecutorLookup` | `(nodeType, executorKey) → Executor?` — how the engine finds an executor without importing the registry. Returns `undefined` if the node doesn't support the mode. |

### Serialization — `serialization.ts` (Layer 1)

| Symbol | What it is |
|--------|-----------|
| `SerializedGraph` | The on-disk / on-the-wire format (version `'1.0'`), with nested `subgraph` support. **The single contract between frontend and backend.** |
| `SerializedGraphData`, `SerializedNode` | The nested payload + per-node shapes inside a `SerializedGraph`. |

---

## Backend (`backend/`, mirrors the engine with real PyTorch)

The backend does the *same* topological walk as the frontend engine but builds
real `nn.Module`s and pushes tensors through them. The two halves meet only at the
`SerializedGraph` JSON contract. Each `type → behavior` boundary is a documented
`typing.Protocol` whose `__call__` shows on hover at every registry call site (a
project convention — see `CLAUDE.md`).

### Engine / graph build — `engine/`

| Symbol | What it is |
|--------|-----------|
| `TorchModuleBuilder` (Protocol) | Builds the *fresh, randomly-initialized* `nn.Module` for one node type from its props + input shapes. Registered per type in `NODE_BUILDERS` (`node_builders.py`). |
| `NodeExecutor` (Protocol) | Runs one node: gathers inputs, gets its module, runs it, stores raw outputs, returns an `Execution`. Registered in `_EXECUTORS`, dispatched by `execute` (`graph_builder/runners.py`). Does *not* build display metadata — that's the `describe_*` half. |
| `Execution` (dataclass) | The raw result of running one node (`kind`, `module`, `outputs`, `primary`, `extra`) — shared by every walk before a presenter formats it. |
| `RunContext` (dataclass) | Everything an executor needs, plus the knobs that differ between walks (`use_trained`, `batch_override`, `skip_losses`). |
| `TensorInfo`, `WeightInfo`, `ActivationInfo`, `GradientInfo`, `BatchNormInfo`, `HistStats` (TypedDicts) | Exact dict shapes the stat helpers emit for the UI (`graph_builder/stats.py`). |

### Data — `dataprep/data_loaders.py`

| Symbol | What it is |
|--------|-----------|
| `BatchLoader` (Protocol) | Loads *one* batch for shape inference / a forward pass. Registered per dataset in `DATA_LOADERS`. |
| `DatasetFactory` (Protocol) | Builds the full un-batched `Dataset` for training. Registered in `TRAIN_DATASETS` / `TEST_DATASETS`. |
| `Denormalizer` (Protocol) | Undoes a dataset's normalization on one image, for pixel preview. Registered in `DENORMALIZERS`. |
| `DatasetDetailFn` (Protocol) | Returns UI metadata (labels, sample images, stats) for a dataset. Registered in `DATASET_DETAILS`. |
| `CLASS_NAMES` | Dataset type → human-readable class label list. |

### Training — `training/`

| Symbol | What it is |
|--------|-----------|
| `TrainingLoop` (Protocol) | Runs one paradigm end-to-end: `(ctx) → TrainingResult`. Registered per paradigm in `TRAINING_LOOPS` (standard / gan / diffusion / autoregressive). |
| `TrainingContext` (dataclass) | Everything a loop needs — built once by `build_training_context`: parsed graph, built modules, key nodes, hyperparameters, data loaders, tracked samples, callbacks. |
| `TrainingResult` (dataclass) | What every loop returns: per-epoch results, final modules, node results, confusion/misclassification data, mode, error. |
| `EpochMessage` (TypedDict) | One epoch's streamed metrics — the WebSocket `epoch` wire format. **Mirrored by `EpochData` in `src/ui/dashboard/types.ts` — keep in sync.** |
| `detect_training_mode`, `build_training_context`, `run_training` | Paradigm auto-detection, context setup, and the dispatch entry point (drop-in for `train_graph`). See [`../explanation/training-plugins.md`](../explanation/training-plugins.md). |

### Visualization — `visualize/node_viz.py`

| Symbol | What it is |
|--------|-----------|
| `ForwardVizFn` (Protocol) | Computes a node's forward-pass viz payload (`{"transformation": ...}`) for the step-through UI. Registered per type in `FORWARD_VIZ`; backward counterparts in `BACKWARD_VIZ`. Both have a default shape-based fallback. |

---

## The frontend ↔ backend seam

| Contract | Frontend side | Backend side |
|----------|--------------|--------------|
| Graph format | `SerializedGraph` (`serialization.ts`) | `graph_data` dict consumed by the graph builder |
| Streamed epoch metrics | `EpochData` (`ui/dashboard/types.ts`) | `EpochMessage` (`training/base.py`) |
| Node display data | `ExecutionResult.metadata` | `describe_*` presenters → `node_result` |

Nothing else crosses the wire — see the "Frontend ↔ Backend communication" section
of `CLAUDE.md` for the transport (WebSocket for training, REST for the rest).
