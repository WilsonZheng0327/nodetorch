# Weight and Activation Visualization

How NodeTorch extracts, transmits, and renders weight/activation statistics so users can inspect what their model is doing after a forward pass.

## Backend: Extracting Statistics

Two functions in `backend/graph_builder.py` produce visualization data: `module_weight_info()` for learned parameters, and `activation_info()` for layer outputs.

### `module_weight_info(module)`

Called on every layer node that has parameters (Conv2d, Linear, BatchNorm, etc.). Returns `None` for parameter-free layers (ReLU, Flatten, Dropout, pooling).

Steps:

1. Collect all parameter tensors from the module (`module.parameters()`).
2. Concatenate them into a single flat 1D tensor: `torch.cat([p.detach().flatten() for p in params]).float()`.
3. Compute a 30-bin histogram with `torch.histogram(all_weights.cpu(), bins=30)`.
4. Return a dict with summary statistics and histogram data:

```python
{
    "mean": float,       # average weight value
    "std": float,        # standard deviation
    "min": float,        # minimum weight
    "max": float,        # maximum weight
    "histBins": [...],   # 30 left bin edges (from hist.bin_edges[:-1])
    "histCounts": [...], # 30 integer counts per bin
}
```

All floats pass through `_safe_float()`, which converts `NaN`/`Inf` to `None` for JSON safety.

### `activation_info(tensor)`

Called on the output tensor of every layer node, regardless of whether the layer has parameters. This means even ReLU, pooling, and reshape layers produce activation stats.

Steps:

1. Flatten and detach the output tensor: `tensor.detach().flatten().float()`.
2. Compute the same 30-bin histogram: `torch.histogram(flat.cpu(), bins=30)`.
3. Calculate sparsity: `float((flat == 0).sum()) / flat.numel()` -- the fraction of elements that are exactly zero.
4. Return:

```python
{
    "mean": float,
    "std": float,
    "min": float,
    "max": float,
    "histBins": [...],      # 30 left bin edges
    "histCounts": [...],    # 30 counts
    "sparsity": float,      # 0.0 to 1.0
}
```

### Where statistics are attached to node results

In `build_and_run_graph()`, the layer node handling section (the final branch, after data/optimizer/loss/structural/subgraph nodes are handled) builds the metadata dict for each layer node:

```python
meta = {
    "outputShape": list(raw_output.shape),
    "paramCount": sum(p.numel() for p in module.parameters()),
}
wi = module_weight_info(module)
if wi:
    meta["weights"] = wi
meta["activations"] = activation_info(raw_output)
```

This means:
- `metadata.weights` is only present when the module has trainable parameters.
- `metadata.activations` is always present on layer nodes.
- Data nodes, loss nodes, optimizer nodes, and structural nodes do not get weight/activation data.

The same pattern applies to multi-output nodes (LSTM/GRU), where `activation_info` is called on the first output tensor.

## Frontend: Rendering in PropertyInspector

`src/ui/PropertyInspector.tsx` reads `metadata.weights` and `metadata.activations` from the selected node's `lastResult` and renders two collapsible sections.

### Weights section

Rendered when `metadata?.weights` is truthy:

- **Mean** -- `metadata.weights.mean` formatted to 4 decimal places
- **Std** -- `metadata.weights.std` formatted to 4 decimal places
- **Range** -- `metadata.weights.min` to `metadata.weights.max`, both to 4 decimal places
- **Histogram** -- a `<Histogram>` component with blue color (`#89b4fa`), labeled "Weight distribution"

### Activations section

Rendered when `metadata?.activations` is truthy:

- **Mean** -- `metadata.activations.mean` to 4 decimal places
- **Std** -- `metadata.activations.std` to 4 decimal places
- **Sparsity** -- `metadata.activations.sparsity` displayed as a percentage (e.g. "45.2%"), or a dash if null
- **Histogram** -- a `<Histogram>` component with green color (`#10b981`), labeled "Activation distribution"

### The Histogram component

`Histogram` is a canvas-based bar chart defined at the bottom of `PropertyInspector.tsx`. It takes `bins` (30 left edges), `counts` (30 integers), a `color`, and a `label`.

Rendering steps:

1. Get the canvas's bounding rect and scale for `devicePixelRatio` (retina support).
2. Clear the canvas.
3. Find `maxCount = Math.max(...counts)` for normalization.
4. Calculate `barWidth = canvasWidth / counts.length` (equal-width bars filling the canvas).
5. For each of the 30 bins, draw a filled rectangle:
   - x position: `i * barWidth`
   - Height: `(counts[i] / maxCount) * (canvasHeight - 2)`, proportional to the maximum bin
   - y position: bottom-aligned (`canvasHeight - barHeight`)
   - Width: `barWidth - 1` (1px gap between bars)
6. Fill color is set to the provided `color` at 60% opacity (`globalAlpha = 0.6`).

The canvas is styled at 100% width and 50px height, with a dark background (`#313244`) and rounded corners.

## What sparsity means

Sparsity measures the fraction of activation values that are exactly zero. This is especially meaningful for ReLU and its variants:

- **ReLU** clamps all negative values to zero, so sparsity directly shows what percentage of neurons are "dead" (not firing) for the current input.
- A sparsity of 0% means every neuron fired (all activations positive).
- A sparsity of 50% means half the neurons produced zero output.
- Very high sparsity (e.g. 95%+) could indicate a dying ReLU problem where most neurons never activate, which is a signal the learning rate may be too high or initialization is poor.

For layers without ReLU (e.g. a raw Conv2d output before activation), sparsity is typically near 0% since it is unlikely for convolution outputs to land exactly on zero.

## End-to-end trace: Conv2d followed by ReLU

Suppose the graph is: **MNIST** -> **Conv2d** (1->16, kernel 3) -> **ReLU** -> ... and the user clicks **Run** (forward pass).

### 1. Frontend sends the graph

`useGraph.runForward()` serializes the full graph via `serializeGraph()` and POSTs it to `http://localhost:8000/forward`.

### 2. Backend executes the graph

`execute_graph()` calls `build_and_run_graph()` inside `torch.no_grad()`:

- **MNIST node**: The data loader returns a batch tensor of shape `[batchSize, 1, 28, 28]` and a labels tensor. Stored in results. No weight/activation info.
- **Conv2d node**: The builder creates an `nn.Conv2d(1, 16, 3)` module. The MNIST output is fed through it, producing a tensor of shape `[batchSize, 16, 26, 26]`. Then:
  - `module_weight_info(conv_module)` concatenates the weight tensor `[16, 1, 3, 3]` (144 values) and the bias tensor `[16]` (16 values) into a flat tensor of 160 values, computes a 30-bin histogram over those 160 values, and returns the statistics.
  - `activation_info(conv_output)` flattens the `[batchSize, 16, 26, 26]` output into `batchSize * 10816` values, computes a 30-bin histogram, and calculates sparsity (likely near 0% since raw convolution outputs are rarely exactly zero).
  - Both are attached to `metadata.weights` and `metadata.activations`.
- **ReLU node**: The builder creates an `nn.ReLU()` module (no parameters). The Conv2d output is fed through, clamping negatives to zero. Then:
  - `module_weight_info(relu_module)` returns `None` (no parameters), so `metadata.weights` is not set.
  - `activation_info(relu_output)` computes stats on the post-ReLU tensor. The histogram will show a spike at zero (from clamped negatives) plus a spread of positive values. Sparsity might be around 30-60% depending on the data and initialization.

### 3. Backend returns results

`execute_graph()` returns a dict keyed by node ID. Each entry has `outputs` (tensor info) and `metadata` (shapes, param counts, weights, activations). The FastAPI handler wraps this in `{"status": "ok", "results": ...}`.

### 4. Frontend applies results

Back in `useGraph.runForward()`, for each node result:
- The node's `lastResult` is updated with the metadata from the backend.
- `syncToRF()` triggers React re-render.

### 5. User selects Conv2d node

The PropertyInspector renders:
- **Info section**: Output shape `[4, 16, 26, 26]`, Parameters `160`
- **Weights section**: Mean (near 0, from Kaiming init), Std (near 0.4), Range, and a blue histogram showing the weight distribution -- typically a roughly normal curve centered near zero for a freshly initialized Conv2d.
- **Activations section**: Mean, Std, Sparsity (low, near 0%), and a green histogram showing the activation distribution -- a spread of positive and negative values since no ReLU has been applied yet.

### 6. User selects ReLU node

The PropertyInspector renders:
- **Info section**: Output shape `[4, 16, 26, 26]`, Parameters `0`
- **No Weights section** (ReLU has no parameters)
- **Activations section**: Mean (positive, higher than Conv2d's mean since negatives are gone), Std, Sparsity (e.g. "42.3%" -- a significant chunk of values are now zero), and a green histogram with a tall bar at the zero bin and a tail of positive values.

This gives the student a clear visual: the Conv2d layer learned some filters (inspect their weight distribution), and the ReLU killed off the negative activations (visible in the sparsity metric and the histogram's zero spike).
