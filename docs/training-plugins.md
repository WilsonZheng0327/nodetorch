# Training Loop Plugin System

NodeTorch supports multiple training paradigms (standard, GAN, diffusion) through a plugin system. Each paradigm is a self-contained training function that receives shared context and returns results in a standard format.

## Architecture Overview

```
graph_builder.py                  training/
  train_graph() ──lazy import──→   __init__.py
                                     ├── run_training()      ← entry point
                                     ├── detect_training_mode()
                                     ├── TRAINING_LOOPS = {  ← registry
                                     │     "standard": standard_train,
                                     │     "gan": gan_train,
                                     │     "diffusion": diffusion_train,
                                     │   }
                                     │
                                     ├── base.py             ← shared infrastructure
                                     │     TrainingContext
                                     │     TrainingResult
                                     │     build_training_context()
                                     │     save_training_results()
                                     │     build_optimizer()
                                     │     build_scheduler()
                                     │     run_validation_pass()
                                     │     collect_node_snapshots()
                                     │     ...
                                     │
                                     ├── standard.py         ← forward → loss → backward
                                     ├── gan.py              ← alternating G/D updates
                                     └── diffusion.py        ← noise-conditioned denoising
```

## How Training is Triggered

1. User clicks "Train" in the frontend toolbar
2. WebSocket sends `{ type: "train", graph: {...} }` to the backend (`main.py` line ~474)
3. `main.py` calls `train_graph()` in a thread executor
4. `train_graph()` (`graph_builder.py`) is a thin wrapper:

```python
# graph_builder.py
def train_graph(graph_data, on_epoch=None, on_batch=None, cancel_event=None):
    from training import run_training  # lazy import avoids circular deps
    return run_training(graph_data, on_epoch, on_batch, cancel_event)
```

5. `run_training()` (`training/__init__.py`) orchestrates:

```python
# training/__init__.py
def run_training(graph_data, on_epoch, on_batch, cancel_event):
    # Phase 1: Build shared context (modules, data, optimizers, etc.)
    ctx = build_training_context(graph_data, on_epoch, on_batch, cancel_event)

    # Phase 2: Auto-detect which training paradigm to use
    mode = detect_training_mode(ctx.nodes)  # "standard", "gan", or "diffusion"

    # Phase 3: Run the appropriate training loop
    loop_fn = TRAINING_LOOPS[mode]  # e.g., standard_train
    result = loop_fn(ctx)

    # Phase 4: Store results (model weights, confusion matrix, etc.)
    return save_training_results(ctx, result)
```

## TrainingContext — Shared State

`build_training_context()` (`training/base.py` line 112) does all the setup that's common to every paradigm:

- Finds optimizer, data, and loss nodes in the graph
- Seeds random number generators for reproducibility
- Calls `build_and_run_graph()` to build `nn.Module` instances (initial forward pass determines input shapes)
- Loads the dataset, creates train/val DataLoaders
- Picks tracked samples for per-epoch probing

```python
@dataclass
class TrainingContext:
    graph_data: dict                    # raw graph JSON (for run history)
    nodes: dict                         # node_id → node dict
    edges: list                         # edge list
    order: list[str]                    # topological order
    modules: dict[str, nn.Module]       # node_id → built PyTorch module
    initial_node_results: dict          # shape inference results

    data_node: dict                     # the dataset node
    data_node_id: str                   # its ID
    loss_node_ids: list[str]            # list (GAN has multiple)
    optimizer_nodes: list[dict]         # list (GAN has two)
    dataset_type: str                   # e.g. "data.mnist"

    epochs: int                         # from primary optimizer properties
    batch_size: int
    grad_clip_norm: float
    early_stop_patience: int
    val_split: float

    train_loader: DataLoader
    val_loader: DataLoader | None
    train_dataset: Dataset              # for tracked sample picking
    data_loader_fn: callable            # for final forward pass
    tracked_samples: list[dict]         # fixed samples probed each epoch

    on_epoch: callable | None           # streams epoch data to WebSocket
    on_batch: callable | None           # streams batch progress
    cancel_event: threading.Event | None
```

Every training loop receives this context and can access anything it needs without duplicating setup logic.

## TrainingResult — What Loops Return

```python
@dataclass
class TrainingResult:
    epoch_results: list[dict]           # one dict per epoch (streamed to frontend)
    modules: dict[str, nn.Module]       # trained modules (stored in _model_store)
    node_results: dict                  # per-node display metadata
    final_results: dict                 # final forward pass tensors
    confusion_data: dict | None         # confusion matrix (classification only)
    misclassifications: list | None     # misclassified samples
    training_mode: str                  # "standard", "gan", "diffusion"
    error: str | None                   # set if training failed
```

`save_training_results()` (`training/base.py`) takes the result and:
- Stores trained modules in `_model_store["current"]` (for inference)
- Caches results in `_last_run` (for layer detail queries)
- Saves run history to disk (for the Runs tab)
- Sets optimizer node metadata (final loss/accuracy shown on the node)

## Shared Utilities (training/base.py)

These functions are used by all training loops:

| Function | Purpose | Used by |
|----------|---------|---------|
| `build_optimizer(node, params)` | Creates Adam/AdamW/SGD from node properties | standard, GAN |
| `build_scheduler(opt, type, epochs)` | Creates LR scheduler (cosine, step, warmup) | standard, diffusion |
| `init_weight_norms(modules)` | Baseline weight norms for delta tracking | standard |
| `compute_batch_accuracy(results, ...)` | Classification accuracy from predictions | standard |
| `run_validation_pass(ctx, modules, loss_id)` | Validation loop using `run_forward_pass` | standard |
| `collect_node_snapshots(modules, ...)` | Per-node weight/gradient/activation histograms | standard, GAN, diffusion |
| `build_gradient_flow(snapshots, ...)` | Gradient norm per trainable layer | standard, GAN, diffusion |
| `build_epoch_result(epoch, ctx, ...)` | Assembles the epoch dict for WebSocket streaming | standard |
| `run_final_forward(ctx, modules)` | Post-training forward for display metadata | standard, GAN, diffusion |
| `save_training_results(ctx, result)` | Stores in _model_store, _last_run, run history | all |

## run_forward_pass — The Core Forward Dispatch

`forward_utils.py` provides `run_forward_pass()` which runs the full graph forward in topological order:

```python
def run_forward_pass(modules, nodes, edges, order, data_inputs):
    """
    data_inputs: pre-filled results for data/noise nodes
                 e.g., {data_nid: {"out": images, "labels": labels}}

    Walks nodes in topological order. For each node:
    - Skip if already in data_inputs (data nodes, noise nodes)
    - Skip optimizer nodes
    - Dispatch via execute_node() which handles:
        - LOSS_NODES: module(predictions, labels)
        - MULTI_INPUT_NODES: module(**kwargs)
        - SUBGRAPH_TYPE: module(**inputs), take first output
        - Default: module(inputs["in"])
    """
```

This replaces the ~30-line dispatch block that was previously duplicated 12 times across `train_graph`, `evaluate_test_set`, `infer_graph`, `_probe_tracked_samples`, and others.

## How Each Training Mode Works

### Standard Mode (`training/standard.py`)

Detected when: no GAN or diffusion nodes present (the default).

```
for each epoch:
    for each batch (images, labels):
        forward pass through entire graph
        loss.backward()
        optimizer.step()
        compute accuracy (classification) or skip (reconstruction)
    validation pass
    early stopping check
    collect snapshots, gradient flow, tracked sample probes
    stream epoch result to frontend
```

Handles both classification (CrossEntropy — reports accuracy, confusion matrix) and reconstruction (MSE/VAE — reports loss only). The distinction is automatic: `compute_batch_accuracy` checks if predictions are 2D `[B, classes]`.

### GAN Mode (`training/gan.py`)

Detected when: graph contains `ml.gan.noise_input` or `ml.loss.gan`.

The GAN training loop doesn't use the standard `run_forward_pass` because the discriminator must run twice per batch (on real and fake data):

```
Identify G and D by graph structure:
  - Generator = subgraph block downstream of noise_input
  - Discriminator = other subgraph block(s)

for each epoch:
    for each batch (real_images, _):
        noise = randn(batch_size, latent_dim)

        # ---- Train Discriminator ----
        freeze generator
        fake_images = generator(noise)               # no grad for G
        real_scores = discriminator(real_images)
        fake_scores = discriminator(fake_images.detach())
        d_loss = gan_loss(real_scores, fake_scores)   # BCE with label smoothing
        d_loss.backward()
        optimizer_d.step()

        # ---- Train Generator ----
        freeze discriminator
        fake_images = generator(noise)                # fresh forward
        fake_scores = discriminator(fake_images)      # grad flows through D to G
        g_loss = BCE(fake_scores, ones)               # G wants D to think fakes are real
        g_loss.backward()
        optimizer_g.step()

    epoch result: dLoss, gLoss, generated sample images
```

Key differences from standard:
- Two optimizers, two backward passes per batch
- Discriminator runs twice — not a DAG constraint, handled programmatically
- Generator loss is computed directly (not through the GAN loss node)
- No accuracy/confusion matrix — replaced by D loss and G loss
- Generates sample images every 5 epochs for visualization

### Diffusion Mode (`training/diffusion.py`)

Detected when: graph contains `ml.diffusion.noise_scheduler`.

```
Find noise scheduler module and its alpha_cumprod schedule

for each epoch:
    for each batch (clean_images, _):
        t = randint(0, num_timesteps, (batch_size,))   # random timesteps
        noise = randn_like(clean_images)                # actual noise
        noisy = scheduler.add_noise(clean_images, noise, t)  # x_t = sqrt(a)*x + sqrt(1-a)*noise
        t_channel = (t / T).expand_as(noisy)            # timestep as spatial channel
        model_input = cat([noisy, t_channel], dim=1)     # [B, C+1, H, W]

        predicted_noise = model(model_input)             # forward through conv layers
        loss = MSE(predicted_noise, actual_noise)
        loss.backward()
        optimizer.step()

    every 5 epochs: run full DDPM sampling loop to generate preview images
```

The noise scheduler node has two output ports:
- `out`: noisy images with timestep channel `[B, C+1, H, W]` → feeds into the model
- `noise`: actual noise `[B, C, H, W]` → feeds into MSE loss as the target

Key differences:
- No classification accuracy — only noise prediction loss
- Timestep conditioning via extra image channel (simplest approach)
- DDPM sampling for visualization (iterative denoising from pure noise)

## How Auto-Detection Works

`detect_training_mode()` (`training/__init__.py` line 27) scans node types:

```python
def detect_training_mode(nodes):
    for n in nodes.values():
        if n["type"] in ("ml.gan.noise_input", "ml.loss.gan"):
            return "gan"
        if n["type"] == "ml.diffusion.noise_scheduler":
            return "diffusion"
    return "standard"
```

This runs after `build_training_context()`, so the graph is already parsed. Detection is based purely on which node types are present — no user configuration needed.

## Epoch Result Streaming

Each training loop calls `ctx.on_epoch(epoch_result)` after every epoch. The WebSocket handler in `main.py` picks this up and streams it to the frontend in real-time.

Standard epoch result:
```json
{
    "epoch": 1, "totalEpochs": 10,
    "loss": 0.523, "accuracy": 0.82,
    "valLoss": 0.61, "valAccuracy": 0.79,
    "learningRate": 0.001, "time": 12.3,
    "gradientFlow": [{"name": "conv2d", "norm": 0.042}, ...],
    "perClassAccuracy": [{"cls": 0, "accuracy": 0.95}, ...],
    "nodeSnapshots": { "node_id": {"weights": {...}, "gradients": {...}} },
    "trackedSamples": [{"idx": 42, "label": 3, "confidence": 0.87, ...}]
}
```

GAN epoch result:
```json
{
    "epoch": 1, "totalEpochs": 50,
    "dLoss": 0.58, "gLoss": 2.34,
    "loss": 0.58, "accuracy": 0.0,
    "learningRate": 0.0002, "time": 8.1,
    "generatedSamples": [{"pixels": [[...]], "channels": 1}, ...]
}
```

Diffusion epoch result:
```json
{
    "epoch": 1, "totalEpochs": 20,
    "loss": 0.0234, "accuracy": 0.0,
    "noiseLoss": 0.0234,
    "learningRate": 0.001, "time": 15.2,
    "denoisedSamples": [{"pixels": [[...]], "channels": 1}, ...]
}
```

The frontend training dashboard renders whatever fields are present — missing fields (like `accuracy` for GAN) are gracefully skipped.

## Adding a New Training Paradigm

### Step 1: Create the training loop

Create `backend/training/<name>.py`:

```python
from .base import TrainingContext, TrainingResult, build_optimizer, ...

def my_train(ctx: TrainingContext) -> TrainingResult:
    # Access everything via ctx:
    #   ctx.modules        — built nn.Modules
    #   ctx.train_loader   — DataLoader
    #   ctx.epochs         — number of epochs
    #   ctx.on_epoch       — callback to stream progress
    #   ctx.cancel_event   — check for cancellation

    optimizer = build_optimizer(ctx.optimizer_nodes[0], all_params)
    epoch_results = []

    for epoch in range(ctx.epochs):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            break

        # ... your training logic ...

        epoch_result = {"epoch": epoch + 1, "totalEpochs": ctx.epochs, "loss": avg_loss, ...}
        epoch_results.append(epoch_result)
        if ctx.on_epoch:
            ctx.on_epoch(epoch_result)

    final_results, node_results = run_final_forward(ctx, ctx.modules)

    return TrainingResult(
        epoch_results=epoch_results,
        modules=ctx.modules,
        node_results=node_results,
        final_results=final_results,
        training_mode="my_mode",
    )
```

### Step 2: Register it

In `backend/training/__init__.py`:

```python
from .my_mode import my_train

TRAINING_LOOPS["my_mode"] = my_train
```

### Step 3: Add detection

```python
def detect_training_mode(nodes):
    for n in nodes.values():
        if n["type"] == "ml.my_mode.special_node":
            return "my_mode"
    ...
```

### Step 4: Add node types (if needed)

Follow the existing patterns:
- Frontend: `src/domain/nodes/<category>/<node>.ts` with shape executor
- Backend: builder in `node_builders.py`, registered in `NODE_BUILDERS`
- Viz: optional entry in `node_viz.py`
- Validation: update `src/core/validation.ts`

### Step 5: Create a preset

Add `model-presets/<name>.json` so users can one-click load the architecture.

## File Reference

```
backend/
  graph_builder.py          — train_graph() thin wrapper, constants, utilities
  forward_utils.py          — execute_node(), run_forward_pass()
  training/
    __init__.py             — registry, detect_training_mode(), run_training()
    base.py                 — TrainingContext, TrainingResult, shared utilities
    standard.py             — standard_train() — classification, reconstruction, VAE
    gan.py                  — gan_train() — alternating G/D updates
    diffusion.py            — diffusion_train() — noise-conditioned denoising
```

## Circular Import Prevention

The `training/` package imports from `graph_builder.py` (constants, `build_and_run_graph`, etc.) at module level. `graph_builder.py` imports from `training/` only inside the `train_graph()` function body (lazy import). This breaks the cycle:

```
graph_builder.py  ──(module-level)──→  [no training import]
     ↑
     └──(lazy, inside function body)──  training/__init__.py
                                            ↑
training/base.py  ──(module-level)──→  graph_builder.py  ✓ (no cycle at import time)
```
