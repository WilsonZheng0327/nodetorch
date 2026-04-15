# Training Pipeline: End-to-End Flow

How a training run works from the moment the user clicks Train to the final results appearing on the canvas.

## Overview

```
User clicks Train
  -> validateTraining() checks graph structure
  -> WebSocket opens to ws://localhost:8000/ws
  -> Serialized graph sent as { type: "train", graph: ... }
  -> Backend spawns train_graph() in a thread executor
  -> Each epoch: callback -> asyncio.Queue -> WebSocket -> frontend progress update
  -> After all epochs: final forward pass, model stored, results sent back
  -> Frontend applies results to nodes, shows training dashboard
```

## Step 1: Frontend validation

When the user clicks Train, `useGraph.runTrain()` is called (`src/ui/useGraph.ts`, line 642).

First, `validateTraining()` from `src/core/validation.ts` performs pre-flight checks. This includes all forward-pass checks plus training-specific requirements:

1. **Forward checks**: Graph is non-empty, has edges, every non-optional input port is connected.
2. **Data node**: Exactly one dataset node (e.g. `data.mnist`) must exist.
3. **Loss node**: At least one loss node (`ml.loss.cross_entropy` or `ml.loss.mse`) with both `predictions` and `labels` ports connected.
4. **Optimizer node**: Exactly one optimizer node (`ml.optimizers.sgd`, `ml.optimizers.adam`, or `ml.optimizers.adamw`) with its `loss` port connected.
5. **Path check**: A BFS from the data node must reach the loss node, ensuring the model layers actually connect the two.

If any check fails, the errors are displayed in the status bar and training does not start.

## Step 2: WebSocket connection

If validation passes, the frontend:

1. Sets status to `{ type: 'running', message: 'Training...' }`.
2. Clears any previous `trainingProgress` data.
3. Serializes the full graph with `serializeGraph()`.
4. Opens a WebSocket to `ws://localhost:8000/ws`.
5. On `ws.onopen`, sends: `{ type: "train", graph: { version: "1.0", graph: { nodes, edges } } }`.

The WebSocket is stored in `trainWsRef` so the cancel button can access it.

## Step 3: Backend receives the train message

In `backend/main.py`, the `websocket_endpoint` handler (`/ws`) runs an async message loop.

When a message with `type: "train"` arrives:

1. A `threading.Event` called `cancel_event` is created.
2. An `asyncio.Queue` called `epoch_queue` is created for streaming results.
3. An `on_epoch` callback is defined that puts epoch data into `epoch_queue`.
4. `train_graph()` is dispatched to a thread executor via `loop.run_in_executor()`, receiving the graph data, the `on_epoch` callback, and the `cancel_event`.

The handler then enters a polling loop:
- Every 0.3 seconds (via `asyncio.wait_for` with timeout), it checks `epoch_queue` for new epoch data and sends it over the WebSocket as `{ type: "epoch", epoch, totalEpochs, loss, accuracy, time, batches, samples }`.
- Between polls, it drains `msg_queue` looking for `cancel` or `_disconnect` messages.
- The loop runs until `train_task.done()` is true.

After the training task completes, any remaining epoch results are drained from the queue and sent, then the final result is sent as `{ type: "train_result", status: "ok", results: {...}, cancelled: bool }`.

## Step 4: Backend training loop

`train_graph()` in `backend/graph_builder.py` (line 423) orchestrates the actual PyTorch training.

### Setup phase

1. **Find the optimizer node**: Scans all nodes for a type in `OPTIMIZER_NODES`. Extracts hyperparameters: `epochs`, `lr`, `momentum`, `weight_decay`, and Adam-specific `beta1`/`beta2`.
2. **Find the data node**: Scans for a node type in `DATA_LOADERS`. Extracts `batch_size`.
3. **Initial forward pass**: Calls `build_and_run_graph(graph_data)` which:
   - Topologically sorts all nodes.
   - For each node in order: creates an `nn.Module` via the appropriate builder in `NODE_BUILDERS`, runs the module on its inputs, and stores the output tensor.
   - Returns: `modules` (dict of node_id -> nn.Module), `results` (dict of node_id -> tensors), `node_results` (per-node metadata), `nodes`, `edges`.

   This initial pass establishes all module shapes -- e.g., `nn.Linear` needs to know its input size, which depends on the upstream Conv2d + Flatten output.

4. **Collect parameters**: Iterates over all modules and gathers `module.parameters()` into a flat list. If no trainable parameters exist, returns an error.
5. **Create the optimizer**: Instantiates `torch.optim.SGD`, `torch.optim.Adam`, or `torch.optim.AdamW` with the collected parameters and hyperparameters.
6. **Find the loss node**: Scans `nodes` for a type in `LOSS_NODES`.
7. **Load the full training dataset**: Calls the dataset's `TRAIN_DATASETS` builder (which returns a `torch.utils.data.Dataset`), then wraps it in a `DataLoader` with the configured batch size and `shuffle=True`.

### Epoch loop

For each epoch (up to the configured count):

1. **Check cancellation**: If `cancel_event.is_set()`, break out of the epoch loop.
2. **Batch loop**: For each `(images, labels)` batch from the DataLoader:
   - Check cancellation again.
   - `optimizer.zero_grad()` -- clear accumulated gradients.
   - **Forward pass through the graph**: Walk the topological order, feeding each node:
     - Data nodes get the current batch's `images` and `labels`.
     - Optimizer nodes are skipped.
     - Loss nodes receive `predictions` and `labels` from their input edges and compute the loss.
     - Structural nodes (Add, Concat) receive named inputs.
     - Subgraph nodes delegate to their `SubGraphModule.forward()`.
     - Layer nodes receive their single `in` input and produce `out`.
   - **Backward pass**: Retrieve the loss tensor from the loss node's results, call `loss_tensor.backward()`, then `optimizer.step()`.
   - **Accumulate metrics**: Add the batch loss to `total_loss`. Find the node feeding `predictions` to the loss node, compute `argmax` predictions, and count correct classifications.

3. **Epoch callback**: After all batches, compute average loss and accuracy, package into an `epoch_result` dict, and call `on_epoch(epoch_result)`. This puts the data into the `epoch_queue` for the WebSocket handler to pick up.

### Final forward pass

After the epoch loop completes (or is cancelled), a final forward pass runs under `torch.no_grad()` using the now-trained modules:

- Data nodes reload a fresh sample batch.
- Layer nodes run their trained modules and produce updated output shapes and tensor info.
- The optimizer node's metadata gets `finalLoss` and `finalAccuracy` from the last epoch.
- Loss nodes recompute with the trained weights.

### Model storage

The trained modules dict is stored in `_model_store["current"]` (line 713). This in-memory store is what enables inference later -- `infer_graph()` retrieves these same module objects with their trained weights.

### Return value

```python
{
    "nodeResults": { node_id: { "outputs": {...}, "metadata": {...} }, ... },
    "epochs": [ { "epoch": 1, "loss": 2.3, "accuracy": 0.12, ... }, ... ],
}
```

## Step 5: Streaming epoch progress to the frontend

Each time the backend's `on_epoch` callback fires, the data flows:

1. `on_epoch(epoch_data)` calls `epoch_queue.put_nowait(epoch_data)` (non-blocking, thread-safe via asyncio Queue).
2. The async polling loop in the WebSocket handler picks it up within 0.3 seconds.
3. Sends over WebSocket: `{ type: "epoch", epoch: 1, totalEpochs: 10, loss: 2.3, accuracy: 0.12, time: 4.5, batches: 938, samples: 60000 }`.
4. Frontend `ws.onmessage` handler receives it.
5. `setTrainingProgress` appends the new epoch to the progress array.
6. `setStatus` updates the status bar with a formatted string like "Epoch [3/10] -- loss: 0.4532, acc: 87.3% (4.2s)".

The `TrainingDashboard` component (`src/ui/TrainingDashboard.tsx`) renders this progress:

- Auto-opens when `isTraining` becomes true.
- Shows a progress bar (epoch N / totalEpochs).
- Displays current loss, accuracy, and time as summary metrics.
- Renders a canvas-based line chart (switchable between loss and accuracy tabs) that updates as new epochs arrive.
- Shows an epoch-by-epoch table with all metrics.

## Step 6: Cancel flow

### Frontend initiates cancel

When the user clicks Cancel, `useGraph.cancelTrain()` sends `{ type: "cancel" }` over the existing WebSocket connection and sets status to "Cancelling training...".

### Backend receives cancel

The WebSocket handler's polling loop checks `msg_queue` between epoch polls. When it finds a `cancel` message, it calls `cancel_event.set()`.

### Training loop checks the event

The training loop checks `cancel_event.is_set()` at two points:
1. At the start of each epoch (line 525).
2. At the start of each batch within an epoch (line 536).

When the event is set, the loop breaks. The final forward pass still runs, and the results are returned. The WebSocket response includes `cancelled: true`.

### Disconnect handling

If the WebSocket disconnects during training (user closes tab, network drop), the `read_messages` task puts a `_disconnect` sentinel in `msg_queue`. The polling loop sees this and sets `cancel_event`, stopping the training. The `finally` block in the WebSocket handler also sets `cancel_event` as a safety net.

## Step 7: Frontend applies final results

When the `train_result` message arrives:

1. `applyResults()` iterates over `results.nodeResults` and merges each node's metadata into its `lastResult`, preserving any existing metadata and adding `forwardResults`.
2. `syncToRF()` triggers React re-render, causing all nodes to update their displays (shapes, param counts, loss values, etc.).
3. `setModelTrained(true)` and `setModelStale(false)` -- this enables the Infer button and marks the model as current.
4. Status is set to "Training complete" (or "Training cancelled"), which auto-clears after 5 seconds.
5. The WebSocket is closed.

### What the user sees after training

- **Optimizer node**: Displays `finalLoss` and `finalAccuracy` from the last epoch.
- **Loss node**: Shows the recomputed loss value from the final forward pass.
- **Layer nodes**: Show updated output shapes and parameter counts.
- **Training dashboard**: Shows the final loss/accuracy chart, a "Complete" badge, and the full epoch table.
- **Infer button**: Now enabled -- the user can run single-sample inference using the stored trained weights.

### Model staleness

After training completes, any graph mutation (adding/removing nodes, adding/removing edges, changing any property) sets `modelStale = true` via the `invalidateModel()` callback. If the user tries to run inference with a stale model, the frontend rejects it with "Model outdated -- graph changed since last training. Retrain first."
