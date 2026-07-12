# Walkthrough: the training flow (WebSocket **Train**)

> A `/code-explain` output — a **function-level** trace from the browser's `train`
> message to results streaming back. Code is shown in fenced blocks; everything
> else is prose. Big subsystems are noted as boundaries and listed at the end for
> drill-in. Every argument, return, and key variable is shown with its exact shape.

## The scenario (so every shape below is real)

A small **MNIST CNN classifier**:

```text
data.mnist → conv(3×3, 8ch) → relu → flatten → linear(10) → cross_entropy → adam
```

Config: `batchSize=64`; Adam `epochs=3, lr=0.01, valSplit=0.1, seed=42`. Task is image
classification → the **standard** paradigm. Shapes that recur below:

```text
images  : Tensor[64, 1, 28, 28]  float32     # one batch
labels  : Tensor[64]             int64       # class ids 0..9
conv out: Tensor[64, 8, 26, 26]              # 3×3, no pad: 28→26
flatten : Tensor[64, 5408]                   # 8·26·26 = 5408
logits  : Tensor[64, 10]                     # linear(5408 → 10)
loss    : Tensor[] (scalar, has grad_fn)
```

---

## 1. `websocket_endpoint` — api/training.py:59

The long-lived coroutine for one browser tab. On a `train` message it runs the
(blocking) training on a worker thread and streams per-epoch results back.

**Signature**

```text
websocket_endpoint(ws: WebSocket)   # runs for the whole connection
```

**Key variables**

- `msg` — one decoded client message. For training it looks like:
  ```text
  {"type": "train", "graph": {"version": "1.0", "graph": {"nodes": [...], "edges": [...]}}}
  ```
- `cancel_event` — `None` until a train starts, then a `threading.Event`. The only channel the async side has to stop the worker thread.
- `epoch_queue` — an `asyncio.Queue` bridging **worker-thread → async** (the thread can't `await ws.send`).

**What it does** — the training branch sets up two callbacks and offloads the work:

```text
train_task = loop.run_in_executor(
    None, lambda: train_graph(msg["graph"], on_epoch=on_epoch, on_batch=on_batch, cancel_event=ce))
```

`on_epoch` / `on_batch` are closures handed *into* the training code; each does `epoch_queue.put_nowait((kind, data))`. `run_in_executor(None, …)` runs `train_graph` off the event loop. Then a loop drains `epoch_queue` and forwards each item to the browser:

```text
msg_type, data = await asyncio.wait_for(epoch_queue.get(), timeout=0.3)
await ws.send_text(json.dumps({"type": msg_type, **data}))
```

The `0.3s` timeout is what lets the same loop also poll `msg_queue` for a `cancel`/`_disconnect` and call `cancel_event.set()`. When the thread finishes, `train_task.result()` yields the return of §3 — `{"nodeResults": {...}}` or `{"error": "..."}` — sent as a terminal `train_result` message.

---

## 2. `train_graph` — engine/graph_builder/__init__.py:43

A thin delegator (kept so old call sites can still import from the engine).

**Signature**

```text
train_graph(graph_data: SerializedGraph, on_epoch=None, on_batch=None, cancel_event=None) -> dict
```

**What it does** — nothing but forward the call:

```text
from training import run_training
return run_training(graph_data, on_epoch, on_batch, cancel_event)
```

---

## 3. `run_training` — training/__init__.py:67

The orchestrator: set up → pick paradigm → run → save.

**Signature**

```text
run_training(graph_data: SerializedGraph, on_epoch, on_batch, cancel_event) -> dict
```

**What it does**

```text
ctx = build_training_context(graph_data, on_epoch, on_batch, cancel_event)
if isinstance(ctx, dict) and "error" in ctx:
    return ctx
mode   = detect_training_mode(ctx.nodes)     # -> "standard"
result = TRAINING_LOOPS.get(mode, standard_train)(ctx)
return save_training_results(ctx, result)
```

`build_training_context` (§4) returns either a `TrainingContext` or, on a setup failure (no optimizer / data / loss / params), an `{"error": ...}` dict — that `isinstance` check is when the run aborts and the error flows back to §1. `detect_training_mode` (§5) returns `"standard"` here. `result` is a `TrainingResult` (§6). The final return is §7's response dict.

---

## 4. `build_training_context` — training/base.py:152

Does all setup once and freezes it into a `TrainingContext` the loop reads from.

**Signature**

```text
build_training_context(graph_data, on_epoch, on_batch, cancel_event) -> TrainingContext | {"error": ...}
```

**Arguments**

- `graph_data` — the serialized graph. Exactly:
  ```text
  {"version": "1.0",
   "graph": {"nodes": [ {"id":"d1","type":"data.mnist","properties":{"batchSize":64}},
                        {"id":"o1","type":"ml.optimizers.adam",
                         "properties":{"epochs":3,"lr":0.01,"valSplit":0.1,"seed":42}}, ...5 more... ],
             "edges": [ {"source":{"nodeId":"d1","portId":"out"}, "target":{"nodeId":"c1","portId":"in"}}, ... ]}}
  ```
- `on_epoch` / `on_batch` — the §1 callbacks, `(dict) -> None`.
- `cancel_event` — `threading.Event`, checked to stop early.

**What it does** — the load-bearing steps:

```text
optimizer_nodes = [n for n in nodes_list if n["type"] in OPTIMIZER_NODES]
```
Finds the optimizer. `optimizer_nodes` is a 1-element list holding the Adam node; its `properties` supply `epochs=3, val_split=0.1, seed=42`. If empty → returns `{"error": "No optimizer node in graph"}`.

```text
modules, _, _, nodes, edges = build_and_run_graph(graph_data)
```
**[subsystem]** — builds one `nn.Module` per layer node and infers shapes via a dummy forward. Returns, among other things:
```text
modules = {"c1": Conv2d(1,8,3), "r1": ReLU(), "f1": Flatten(),
           "l1": Linear(5408,10), "x1": CrossEntropyLoss()}   # no module for data/optimizer nodes
nodes   = {"d1": {...}, "c1": {...}, ...}                     # now a dict keyed by id (was a list)
```

```text
train_dataset, train_loader, val_loader = load_dataset(data_node, props, batch_size)
```
`load_dataset` calls **[subsystem]** `resolve_train_dataset` (MNIST → a 60,000-sample `Dataset` yielding `(Tensor[1,28,28], int)`), then splits by `val_split=0.1`:
```text
train_loader : DataLoader over 54,000 samples -> (images Tensor[64,1,28,28], labels Tensor[64])
val_loader   : DataLoader over  6,000 samples -> same shapes, not shuffled
```

```text
class_names = resolve_class_names(data_node)   # ["0","1", ..., "9"]
denorm      = resolve_denormalizer(data_node)  # closure: img -> img*0.3081 + 0.1307
tracked_samples = pick_tracked_samples(train_dataset, denorm, n=12, modules=modules, nodes=nodes, ...)
```
`tracked_samples` is a list of 12 fixed examples followed across the run (biased toward ones the fresh model gets wrong), each `{"image": pixels, "label": int, "index": int}`.

**Returns** — a `TrainingContext`, whose payload fields hold:

```text
nodes         = {"d1":{...}, "c1":{...}, ...}   # dict, 7 entries
edges         = [ {...}, ... ]
order         = ["d1","c1","r1","f1","l1","x1"] # topo order (optimizer excluded)
modules       = {"c1":Conv2d, ..., "x1":CrossEntropyLoss}
data_node_id  = "d1"
loss_node_ids = ["x1"]
class_names   = ["0",...,"9"]
denorm        = <closure>
epochs=3  batch_size=64  val_split=0.1  grad_clip_norm=0.0  early_stop_patience=0
train_loader / val_loader / train_dataset
tracked_samples : list, 12 items
```

The `resolve_*` calls are the custom-dataset seam: *custom → build from the node's spec; built-in → the identical old registry lookup*. See [custom-datasets.md](../features/custom-datasets.md).

---

## 5. `detect_training_mode` — training/__init__.py:54

Picks the loop by scanning node types.

**Signature**

```text
detect_training_mode(nodes: dict) -> str    # "gan" | "diffusion" | "autoregressive" | "standard"
```

**What it does** — iterates `nodes.values()`, returning `"gan"` on a GAN node, `"diffusion"` on a noise scheduler, `"autoregressive"` when `resolve_is_lm(n)` is true (a language-model dataset, built-in or a `data.custom` with `task="language_model"`). Our 7 nodes match none → returns `"standard"`.

---

## 6. `standard_train` — training/standard.py:38

The epoch/batch loop: forward → loss → backward → step, then per-epoch metrics.

**Signature**

```text
standard_train(ctx: TrainingContext) -> TrainingResult
```

**Setup**

```text
optimizer    = build_optimizer(adam_node, all_params)   # torch.optim.Adam(lr=0.01, betas=(0.9,0.999))
loss_node_id = "x1"
total_batches = len(ctx.train_loader)                   # ~844  (54000 / 64)
```

**The batch loop** — per batch:

```text
optimizer.zero_grad()
data_inputs   = {ctx.data_node_id: {"out": images, "labels": labels}}
batch_results = run_forward_pass(ctx.modules, nodes, edges, order, data_inputs)   # §6a
loss_tensor   = batch_results["x1"]["out"]
```

`data_inputs` seeds the forward pass at the data node. `batch_results` is a dict keyed by node id — the full per-node output of the graph for this batch:

```text
batch_results = {
  "d1": {"out": Tensor[64,1,28,28], "labels": Tensor[64]},
  "c1": {"out": Tensor[64,8,26,26]},
  "r1": {"out": Tensor[64,8,26,26]},
  "f1": {"out": Tensor[64,5408]},
  "l1": {"out": Tensor[64,10]},         # logits
  "x1": {"out": Tensor[]},              # scalar loss
}
```

Then the update:

```text
loss_tensor.backward()          # [subsystem] torch autograd: fills param.grad on every weight
optimizer.step()                # Adam step; total_loss += loss_tensor.item()
c, t, pcc, pct = compute_batch_accuracy(batch_results, "x1", edges, labels)
```

`compute_batch_accuracy` argmaxes the logits and returns on-device counts: `c` = correct (scalar tensor), `t` = 64 (int), `pcc`/`pct` = per-class hit/total vectors `Tensor[10]`. It returns `(None, 0, None, None)` for non-classification (2-D-logits gate) — so accuracy is simply skipped for autoencoders.

**End of each epoch**

```text
accuracy = correct / total                                    # e.g. 0.94 by epoch 3
val_loss, val_accuracy = run_validation_pass(ctx, modules, "x1")
node_snapshots = collect_node_snapshots(...)                  # [subsystem]
gradient_flow  = build_gradient_flow(node_snapshots, ...)
tracked_probes = probe_tracked_samples(ctx.tracked_samples, modules, ...)
epoch_result   = build_epoch_result(...)                      # an EpochMessage
if ctx.on_epoch: ctx.on_epoch(epoch_result)                   # -> epoch_queue -> ws.send
```

`epoch_result` is the wire record streamed to the dashboard — this call is *when a chart row appears*. It looks like:

```text
{"epoch": 1, "totalEpochs": 3, "loss": 0.21, "accuracy": 0.94,
 "valLoss": 0.18, "valAccuracy": 0.95, "learningRate": 0.01, "time": 3.1,
 "gradientFlow": [{"name":"conv2d","norm":..}, {"name":"linear","norm":..}],
 "perClassAccuracy": [{"cls":5,"accuracy":0.89}, ...],   # worst-first
 "classNames": ["0",...,"9"],
 "nodeSnapshots": {"c1": {...}, "l1": {...}},
 "trackedSamples": [ ...12 probes... ]}
```

**Returns** — after all epochs, one final forward pass builds per-node display metadata and a confusion matrix:

```text
TrainingResult(
  epoch_results = [ {...}, {...}, {...} ],           # list, 3 EpochMessages
  modules       = ctx.modules,                       # the trained modules
  node_results  = {"c1": {...}, ..., "o1": {...}},    # per-node display metadata
  final_results = {node_id: output_tensor, ...},
  confusion_data= {"data": [[..]*10]*10, "size": 10, "classNames": ["0",...,"9"]},
  misclassifications = [...] or None,
  training_mode = "standard")
```

### 6a. `run_forward_pass` — engine/forward_utils.py:71

Runs the graph once in topological order; the heart of every train/val/probe pass.

**Signature**

```text
run_forward_pass(modules, nodes, edges, order, data_inputs) -> {node_id: {port: tensor}}
```

**Arguments** — `data_inputs` is the pre-seeded data node:

```text
{"d1": {"out": Tensor[64,1,28,28], "labels": Tensor[64]}}
```

**What it does** — walks `order`, skipping the data/optimizer/noise nodes and anything already computed. For each remaining node:

```text
inputs = gather_inputs(node_id, edges, batch_results)   # upstream tensors, keyed by TARGET port
result = execute_node(ntype, module, inputs)            # {port: tensor}
```

`gather_inputs` reads each incoming edge and keys the tensor by the target port — so the loss node `x1` receives `{"predictions": Tensor[64,10], "labels": Tensor[64]}` (two edges, two ports). `execute_node` calls the module the right way for its type: loss → `module(predictions, labels)`; a plain layer → `module(inputs["in"])`; multi-input (Add/Concat) → `module(**inputs)`. The return is the `batch_results` shown in §6.

---

## 7. `save_training_results` — training/base.py:620

Persists the trained model and returns the response.

**Signature**

```text
save_training_results(ctx: TrainingContext, result: TrainingResult) -> {"nodeResults": {...}}
```

**What it does**

```text
_model_store["current"] = result.modules      # makes inference work immediately after
snapshot_trained_model(ctx.graph_data)         # best-effort weights-to-disk (survives restart)
_last_run <- modules / results / nodes / edges / confusionMatrix   # what inspector/backprop read
build_run_record(...) ; save_run(...)          # run-history entry (best-effort, logged on fail)
```

It also attaches the final metrics to the optimizer node so the canvas shows them:

```text
result.node_results["o1"] = {"outputs": {},
    "metadata": {"finalLoss": 0.06, "finalAccuracy": 0.94}}
```

**Returns**

```text
{"nodeResults": result.node_results}
```

This flows back up to §1's `train_task.result()` and out as the terminal `train_result` message.

---

## Drill-in candidates

Subsystems this walkthrough summarized instead of expanding — say the word and I'll
give any one its own `/code-explain` walkthrough:

- **`build_and_run_graph`** — engine/graph_builder/forward.py:25 — graph → one `nn.Module` per node + shape inference. Returns `(modules, results, node_results, nodes, edges)`.
- **`resolve_train_dataset` / the resolver layer** — dataprep/resolve.py — node → `Dataset`/metadata; the custom-dataset seam.
- **`loss_tensor.backward()`** — torch autograd — populates `.grad` on every parameter from the scalar loss.
- **`collect_node_snapshots`** — training/base.py:479 — per-node weight/gradient/activation/batchnorm stats (the `nodeSnapshots` payload).
- **`run_final_forward`** — training/base.py:599 — post-training per-node display metadata.

## At-a-glance call tree

```text
WS "train"  (api/training.py)
 └─ run_in_executor → train_graph → run_training                 (§2, §3)
     ├─ build_training_context                                   (§4)
     │   ├─ [build_and_run_graph]  modules + shapes
     │   ├─ load_dataset → [resolve_train_dataset] → DataLoaders
     │   ├─ resolve_class_names / resolve_denormalizer
     │   └─ pick_tracked_samples  (12 fixed examples)
     ├─ detect_training_mode → "standard"                        (§5)
     ├─ standard_train                                           (§6)
     │   FOR each epoch:
     │     FOR each batch:
     │       run_forward_pass → gather_inputs + execute_node     (§6a)
     │       [backward] → optimizer.step → compute_batch_accuracy
     │     run_validation_pass · [collect_node_snapshots] · build_gradient_flow
     │     probe_tracked_samples · build_epoch_result → on_epoch()  ──▶ streamed to UI
     │   (after epochs) [run_final_forward] · confusion matrix
     └─ save_training_results → store model + history            (§7)
          ↳ {"nodeResults": {...}}  ──▶ WS "train_result"
```
