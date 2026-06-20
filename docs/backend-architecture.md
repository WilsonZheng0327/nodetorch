# Backend Architecture

How the NodeTorch backend turns a visual node graph into running PyTorch — and a
line-by-line walkthrough of what happens when you press **Train**.

The backend's whole job is one translation: take a **serialized graph** (JSON
describing nodes + edges) and turn it into **real PyTorch** — build an
`nn.Module` per node, wire them together by following edges in topological
order, and run forward/backward passes. Training, inference, step-through
visualization, and Python export are all built on top of that one translation.

The design mirrors the TypeScript frontend engine: same topological sort, same
"gather inputs → run module → store output" loop — but with real tensors instead
of shape math.

## Package layout

Dependencies flow strictly downward. **Layers below know nothing about the
layers above**, and `engine` knows nothing about visualization.

```
main.py            FastAPI: HTTP endpoints + WebSocket. Thin; just dispatches.
engine/            ML-agnostic execution core
  graph_builder/   THE HEART — graph → modules → run. (a package, see below)
  node_builders.py node type → nn.Module factory (the ML knowledge)
  forward_utils.py shared "execute one node / whole graph" helpers
dataprep/          datasets + tokenization (data_loaders, bpe, tokenizer_preview)
visualize/         step-through, backprop, denoise, latent, activation-max, loss-landscape
  layers/          per-layer viz functions, one file per layer family
generate/          text + GAN sampling
export/            graph → standalone .py script
persistence/       training-run history on disk
training/          training-loop plugin system (standard, gan, diffusion, autoregressive)
```

> Note: the data package is named `dataprep`, not `datasets`, specifically
> because `dataprep/data_loaders.py` does a runtime `from datasets import
> load_dataset` (HuggingFace) — a local `datasets/` package would shadow it.

## The data model

Every request carries a `graph_data` object:

```jsonc
{
  "graph": {
    "name": "My Model",
    "nodes": [
      { "id": "n1", "type": "data.mnist",          "properties": { "batchSize": 64 } },
      { "id": "n2", "type": "ml.layers.conv2d",     "properties": { "outChannels": 32, "kernelSize": 3 } },
      { "id": "n3", "type": "ml.loss.cross_entropy","properties": {} },
      { "id": "n4", "type": "ml.optimizers.adam",   "properties": { "lr": 0.001, "epochs": 5 } }
      // ...
    ],
    "edges": [
      { "source": { "nodeId": "n2", "portId": "out" },
        "target": { "nodeId": "n3", "portId": "predictions" } }
      // ...
    ]
  }
}
```

- **node**: `{ id, type, properties, subgraph? }`. `type` is a string like
  `"ml.layers.conv2d"`. `properties` holds the knobs. `subgraph` exists only on
  composite "block" nodes.
- **edge**: `{ source: {nodeId, portId}, target: {nodeId, portId} }`. Ports are
  *named*: most layers use `"in"`/`"out"`, loss nodes use
  `"predictions"`/`"labels"`, attention uses `"query"`/`"key"`/`"value"`.

Node types are partitioned into sets in
`backend/engine/graph_builder/constants.py` — `LOSS_NODES`, `OPTIMIZER_NODES`,
`MULTI_INPUT_NODES`, and the subgraph sentinels. The engine branches on these
sets to decide *how* to call each node.

## The execution engine (`engine/graph_builder/`)

This is a package split into focused modules. The `__init__.py` re-exports the
full public API, so callers just write `from engine.graph_builder import X`.

| Module | Responsibility |
|---|---|
| `constants.py` | node-type sets + sentinel ids |
| `_state.py` | **shared mutable state**: `_device`, `_model_store`, `_last_run` |
| `stats.py` | tensor/param statistics for the UI (`tensor_info`, `activation_info`, …) |
| `build.py` | `topological_sort`, `gather_inputs`, `SubGraphModule`, `build_modules` |
| `forward.py` | `build_and_run_graph` (the workhorse), `execute_graph` |
| `detail.py` | `get_layer_detail` (inspector modal data) |
| `inference.py` | `infer_graph`, `evaluate_test_set`, tracked-sample helpers |

### Shared state (the part to understand first)

`engine/graph_builder/_state.py` holds three module-level globals that the rest
of the backend reads and mutates **in place**:

- `_device` — the active torch device (`_state.py:22`).
- `_model_store` — `{"current": {node_id: nn.Module}}`. Trained modules kept in
  memory so `/infer` and `/evaluate-test` can reuse them (`_state.py:36`).
- `_last_run` — a cache of the last forward/train pass: `modules`, `results`,
  `nodes`, `edges` (`_state.py:40`).

`_last_run` is the trick that makes the node inspector cheap: opening a node's
detail modal reads this cache instead of re-running the graph. `training/base.py`
imports `_model_store` and `_last_run` as objects and mutates them — so they must
stay single shared instances (they do; verified after the package split).

### `gather_inputs` — how data flows along edges

`engine/graph_builder/build.py:49`. Given a node, it walks every edge targeting
that node, reads the source node's output at the named source port, and returns
`{target_port: tensor}`:

```python
# edge {source: {nodeId: "conv", portId: "out"}, target: {nodeId: "relu", portId: "in"}}
#   →  inputs["in"] = results["conv"]["out"]
def gather_inputs(node_id, edges, results):
    inputs = {}
    dev = get_device()
    for edge in edges:
        if edge["target"]["nodeId"] == node_id:
            src_id   = edge["source"]["nodeId"]
            src_port = edge["source"]["portId"]
            tgt_port = edge["target"]["portId"]
            if src_id in results and src_port in results[src_id]:
                v = results[src_id][src_port]
                if isinstance(v, torch.Tensor) and v.device != dev:
                    v = v.to(dev)          # move onto the active device
                inputs[tgt_port] = v
    return inputs
```

### `build_and_run_graph` — the workhorse

`engine/graph_builder/forward.py:28`. Walks nodes in topological order and, for
each, branches on its type:

| Node kind | What happens |
|---|---|
| data loader | call the loader → `{out, labels}` tensors |
| optimizer | skipped (drives training, not forward) |
| GAN noise / diffusion scheduler / timestep embed | special-cased dummy outputs for shape inference |
| loss | `module(predictions, labels)` |
| multi-input (Add, Concat, Attention…) | `module(**inputs)` |
| subgraph | build + run a `SubGraphModule` |
| normal layer | `module(inputs["in"])`; LSTM/GRU return dicts |

**Key insight:** each layer is *built and run in the same pass*. A `Linear`
can't be constructed until it knows its `in_features`, which only exists once the
upstream node has actually run and produced a tensor. So the loop interleaves
construction and execution. At the end it populates `_last_run`:

```python
# forward.py — end of build_and_run_graph
_last_run.clear()
_last_run["modules"] = modules
_last_run["results"] = results
_last_run["nodes"]   = nodes
_last_run["edges"]   = edges
return modules, results, node_results, nodes, edges
```

`execute_graph` (`forward.py:393`) just wraps this in `torch.no_grad()` for the
`/forward` endpoint and returns `node_results` (per-node display metadata).

## Where the ML knowledge lives

- `engine/node_builders.py` — `NODE_BUILDERS` (`node_builders.py:691`), a
  `{node_type: build_fn}` registry of ~47 entries. Each `build_*(props,
  input_shapes)` returns an `nn.Module`. This is the **only** place that knows
  `"ml.layers.conv2d"` means `nn.Conv2d`.
- `dataprep/data_loaders.py` — parallel registries keyed by dataset type:
  `DATA_LOADERS` (batches for forward, `:548`), `TRAIN_DATASETS`/`TEST_DATASETS`
  (full datasets for training/eval), `DENORMALIZERS` (image previews),
  `CLASS_NAMES`, `DATASET_DETAILS`, `LM_DATASET_TYPES`.

---

# Walkthrough 1: pressing **Forward**

The simplest path — no backend state changes except the `_last_run` cache.

```
Frontend  ── POST /forward { graph } ─────────────►  main.py:118  forward()
                                                          │
                                                          ▼
                                              execute_graph(graph_data)      forward.py:393
                                                          │ no_grad
                                                          ▼
                                              build_and_run_graph(...)       forward.py:28
                                                  topological_sort           build.py:21
                                                  for each node:
                                                    gather_inputs            build.py:49
                                                    NODE_BUILDERS[type](...) node_builders.py
                                                    module(inputs)
                                                    record tensor_info + metadata
                                                  populate _last_run
                                                          │
Frontend  ◄── { status, results } ◄──────────────────────┘
```

The frontend then renders each node's `metadata` (output shape, param count,
weight/activation histograms) directly on the canvas node.

---

# Walkthrough 2: pressing **Train** (the full story)

Training is the most involved path because it is **streamed** (epoch-by-epoch)
over a WebSocket and runs PyTorch on a background thread so the event loop stays
responsive.

```
User clicks Train
  │
  ├─ Frontend: validateTraining(), open ws://localhost:8000/ws
  │            send { type: "train", graph: {...} }
  │
  ▼
main.py  websocket_endpoint            (main.py:600)
  │  reader_task drains client msgs into msg_queue   (so "cancel" can interrupt)
  │  on {type:"train"}:
  │    cancel_event = threading.Event()
  │    train_task = loop.run_in_executor(None, train_graph, ...)   (main.py:665)
  │    while not train_task.done():
  │       drain epoch_queue → ws.send_text({type:"epoch", ...})    (main.py:670)
  │       check msg_queue for "cancel" / "_disconnect"
  │
  ▼ (on the worker thread)
train_graph(graph, on_epoch, on_batch, cancel_event)   (graph_builder/__init__.py:42)
  └─ run_training(...)                                  (training/__init__.py:43)
       1. ctx  = build_training_context(graph_data, ...)   (training/base.py:113)
       2. mode = detect_training_mode(ctx.nodes)           (training/__init__.py:30)
       3. result = TRAINING_LOOPS[mode](ctx)               (e.g. standard.py:39)
       4. return save_training_results(ctx, result)        (training/base.py:667)
```

### Step 1 — the WebSocket handler dispatches to a thread

`backend/main.py:600`. The handler runs an async message loop. A separate
`reader_task` continuously reads client messages into `msg_queue`, so a `cancel`
can arrive *while training is running*. PyTorch is blocking, so training can't run
on the event loop — it goes to a thread executor:

```python
# main.py — inside websocket_endpoint, on {"type": "train"}
cancel_event = threading.Event()
epoch_queue = asyncio.Queue()

def on_epoch(epoch_data):
    epoch_queue.put_nowait(("epoch", epoch_data))      # called from worker thread

train_task = loop.run_in_executor(
    None,
    lambda: train_graph(msg["graph"], on_epoch=on_epoch, on_batch=on_batch, cancel_event=ce),
)

while not train_task.done():
    try:
        msg_type, data = await asyncio.wait_for(epoch_queue.get(), timeout=0.3)
        await ws.send_text(json.dumps({"type": msg_type, **data}))   # stream to browser
    except asyncio.TimeoutError:
        pass
    while not msg_queue.empty():                        # cancellation path
        pending = msg_queue.get_nowait()
        if pending.get("type") in ("cancel", "_disconnect"):
            cancel_event.set()
```

Data path: **training thread → `on_epoch` → `epoch_queue` → WebSocket →
frontend dashboard**, one epoch at a time. Cancellation is a `threading.Event`
the loop checks between batches.

### Step 2 — `build_training_context` does all setup once

`backend/training/base.py:113`. This runs once before any epoch and returns a
`TrainingContext` dataclass (or an `{"error": ...}` dict). It:

1. Finds the optimizer node(s), reads `epochs`, `gradClip`, `valSplit`, `seed`.
2. Seeds all RNGs (torch / cuda / random / numpy) for reproducibility.
3. Finds the data node.
4. **Builds the modules** by calling `build_and_run_graph` (same function the
   forward pass uses):

   ```python
   modules, results, node_results, nodes, edges = build_and_run_graph(graph_data)
   all_params = list(p for m in modules.values() for p in m.parameters())
   if not all_params:
       return {"error": "No trainable parameters in graph"}
   ```

5. Loads the dataset (swapping in a BPE-encoded dataset if a BPE tokenizer node is
   present), splits train/val.
6. Picks 4 fixed **tracked samples** to follow across epochs (for the dashboard's
   "watch these predictions evolve" panel).

### Step 3 — `detect_training_mode` picks the paradigm

`backend/training/__init__.py:30`. A plugin registry chooses the loop by scanning
node types:

```python
def detect_training_mode(nodes):
    for n in nodes.values():
        ntype = n.get("type", "")
        if ntype in ("ml.gan.noise_input", "ml.loss.gan"): return "gan"
        if ntype == "ml.diffusion.noise_scheduler":        return "diffusion"
        if ntype in LM_DATASET_TYPES:                       return "autoregressive"
    return "standard"

TRAINING_LOOPS = {"standard": standard_train, "gan": gan_train,
                  "diffusion": diffusion_train, "autoregressive": autoregressive_train}
```

### Step 4 — the standard loop runs

`backend/training/standard.py:39` is the canonical pattern. The inner loop is the
classic forward → loss → backward → step, using the shared executor from
`engine/forward_utils.py`:

```python
for epoch in range(ctx.epochs):
    if ctx.cancel_event and ctx.cancel_event.is_set():
        break
    for batch_idx, (images, labels) in enumerate(tqdm(ctx.train_loader, ...)):
        images, labels = images.to(dev), labels.to(dev)
        optimizer.zero_grad()

        # Forward: feed the data node's output, run the whole graph
        data_inputs   = {ctx.data_node_id: {"out": images, "labels": labels}}
        batch_results = run_forward_pass(ctx.modules, ctx.nodes, ctx.edges, ctx.order, data_inputs)

        # Backward: grab the tensor at the loss node and propagate
        loss_tensor = batch_results.get(loss_node_id, {}).get("out")
        if loss_tensor is not None:
            loss_tensor.backward()                          # standard.py:100
            if ctx.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(all_params, ctx.grad_clip_norm)
            optimizer.step()                                # standard.py:103
            total_loss += loss_tensor.item()
            # ... accumulate accuracy, confusion matrix, misclassified samples
```

`run_forward_pass` (`engine/forward_utils.py:71`) walks the topological order and
calls `execute_node` for each, which centralizes the per-type dispatch (loss nodes
take named args, multi-input nodes take `**kwargs`, layers take a single `"in"`,
LSTM/GRU return dicts). This is the same dispatch used by validation, step-through,
and activation-max — written once.

After each epoch the loop also:
- runs a validation pass and applies early stopping,
- steps the LR scheduler,
- **collects per-node snapshots** (`collect_node_snapshots`, `base.py:427`):
  weight / gradient / activation histograms for every layer — this is what powers
  the dashboard's gradient-flow chart and per-layer panels,
- re-probes the 4 tracked samples,
- packs everything into an **epoch result dict** (`build_epoch_result`,
  `base.py:524`) and hands it to `on_epoch` — which is what gets streamed to the
  browser.

### Step 5 — `save_training_results` persists everything

`backend/training/base.py:667`. After the last epoch, the trained modules and the
final forward results are stored into the shared state so inference, test
evaluation, and the node inspector all work afterward:

```python
def save_training_results(ctx, result):
    _model_store["current"] = result.modules     # /infer + /evaluate-test reuse this

    _last_run.clear()                            # /layer-detail reads this
    _last_run["modules"] = result.modules
    _last_run["results"] = result.final_results
    _last_run["nodes"]   = ctx.nodes
    _last_run["edges"]   = ctx.edges
    if result.confusion_data:    _last_run["confusionMatrix"]   = result.confusion_data
    if result.misclassifications: _last_run["misclassifications"] = result.misclassifications

    save_run(build_run_record(...))              # write run history to disk (persistence/)
    return {"nodeResults": result.node_results}
```

### Step 6 — final message back to the browser

The handler drains any remaining epoch results, then sends
`{type: "train_result", status, results, cancelled}`. The frontend applies the
final per-node metadata to the canvas and the dashboard shows the completed
training curves.

**The throughline:** training builds modules with the *same*
`build_and_run_graph` the forward pass uses, runs the standard PyTorch loop with
the *shared* `run_forward_pass` executor, and on completion writes `_model_store`
+ `_last_run` so every downstream feature reuses the trained model instead of
recomputing it.

---

# Walkthrough 3: pressing **Infer** (reusing trained state)

This shows why the shared state matters.

```
Frontend ── POST /infer { graph } ──►  main.py:254  infer()
                                            │
                                            ▼
                                   infer_graph(graph_data)     inference.py:201
                                       if not has_trained_model():   return error
                                       trained = get_trained_modules()   # _model_store["current"]
                                       for node in topological order:
                                           data node → load ONE sample (batchSize=1)
                                           else      → run the stored trained module
                                       build prediction { predictedClass, confidence, probabilities }
                                            │
Frontend ◄── { results: { nodeResults, prediction } } ◄──┘
```

No modules are rebuilt — `infer_graph` pulls them straight from `_model_store`,
which `save_training_results` populated. Similarly, clicking a node to open its
inspector hits `POST /layer-detail` → `get_layer_detail` (`detail.py:18`), which
reads the `_last_run` cache and produces weight heatmaps / feature maps /
attention maps / confusion matrices without re-running anything.

---

# Request → endpoint cheat sheet

The frontend's execution modes map directly to endpoints (all in `main.py`):

| User action | Endpoint | Backend entry point |
|---|---|---|
| Shape mode | *(none — pure frontend TS)* | — |
| Forward | `POST /forward` | `execute_graph` |
| Train | `WS /ws` (streamed) | `train_graph` → `run_training` |
| Test | `POST /evaluate-test` | `evaluate_test_set` |
| Infer | `POST /infer` | `infer_graph` |
| Inspect a node | `POST /layer-detail` | `get_layer_detail` (reads `_last_run`) |
| Step through | `POST /step-through` | `visualize/step_through.py` |
| Backprop viz | `POST /backward-step-through` | `visualize/backprop_sim.py` |
| Generate text | `POST /generate-text` | `generate/text_generate.py` |
| Generate GAN images | `POST /gan-generate` | `generate/gan_generate.py` |
| Latent grid (VAE) | `POST /latent-grid` | `visualize/latent_viz.py` |
| Activation max | `POST /activation-max` | `visualize/activation_max.py` |
| Loss landscape | `POST /loss-landscape` | `visualize/loss_landscape.py` |
| Export Python | `POST /export-python` | `export/exporter.py` |

---

# Where to look when adding things

- **New layer type** → add a `build_*` to `engine/node_builders.py` and register
  it in `NODE_BUILDERS`. Add a viz function in `visualize/layers/` if you want
  step-through detail.
- **New dataset** → add loaders to `dataprep/data_loaders.py` (`DATA_LOADERS` +
  `TRAIN_DATASETS`/`TEST_DATASETS` + optional `DENORMALIZERS`/`CLASS_NAMES`).
- **New training paradigm** → add a file in `training/`, implement
  `(ctx) -> TrainingResult`, register it in `TRAINING_LOOPS`, and add detection
  to `detect_training_mode`. See `docs/training-plugins.md`.

## Related docs

- `docs/training-flow.md` — the frontend half of the train flow (validation,
  WebSocket, dashboard).
- `docs/training-plugins.md` — the training-paradigm plugin system in depth.
- `docs/shape-inference.md`, `docs/visualization.md`, `docs/multi-output-nodes.md`,
  `docs/custom-blocks.md`.
