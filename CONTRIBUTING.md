# Contributing — How to Add New Functionality

## Adding a New Node Type

### 1. Frontend (shape inference)

Create a file in the appropriate category folder under `src/domain/nodes/`:

```
src/domain/nodes/
  data/          — dataset nodes (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, IMDb, AG News)
  layers/        — neural network layers (Conv2d, Linear, LSTM, Embedding, etc.)
  activations/   — activation functions (ReLU, Sigmoid, Softmax, GELU, etc.)
  loss/          — loss functions (CrossEntropy, MSE)
  optimizers/    — optimizers (SGD, Adam, AdamW)
  structural/    — graph ops (Add, Concat, Reshape, Permute, SequencePool)
  subgraph/      — reusable block nodes (SubGraph, Input/Output sentinels)
```

The file exports a `NodeDefinition` object:

```typescript
import type { NodeDefinition } from '../../../core/nodedef';

export const myNode: NodeDefinition = {
  type: 'ml.layers.my_node',     // namespaced type string
  version: 1,
  displayName: 'MyNode',
  description: 'What this node does',
  category: ['ML', 'Layers'],    // determines palette grouping

  getProperties: () => [
    // Each property auto-generates a UI widget
    {
      id: 'someParam',
      name: 'Some Parameter',
      type: { kind: 'number', min: 1, integer: true },
      defaultValue: 64,
      affects: 'execution',      // 'execution' | 'ports' | 'both'
    },
  ],

  getPorts: (properties) => [
    // Ports can depend on properties (dynamic ports)
    {
      id: 'in',
      name: 'Input',
      direction: 'input',
      dataType: 'tensor',        // must match a registered data type
      allowMultiple: false,
      optional: false,
    },
    {
      id: 'out',
      name: 'Output',
      direction: 'output',
      dataType: 'tensor',
      allowMultiple: true,
      optional: false,
    },
  ],

  executors: {
    shape: {
      execute: async ({ inputs, properties }) => {
        const input = inputs.in;  // key matches port id
        if (!input) return { outputs: {} };

        // Do shape math here
        const outShape = [input[0], properties.someParam];

        return {
          outputs: { out: outShape },  // key matches output port id
          metadata: {
            outputShape: outShape,
            paramCount: 123,           // optional, shown on node
          },
        };
      },
    },
  },
};
```

Then add it to the folder's `index.ts`:

```typescript
// src/domain/nodes/layers/index.ts
import { myNode } from './my-node';

export const layerNodes: NodeDefinition[] = [
  // ...existing nodes,
  myNode,
];
```

That's it for the frontend. The node appears in the palette and shape inference works.

### 2. Backend — Module builder (`backend/node_builders.py`)

Add a builder function that creates the `nn.Module`:

```python
def build_my_node(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    return nn.MyModule(
        some_param=props["someParam"],
        # Derive input-dependent args from in_shape
    )

NODE_BUILDERS["ml.layers.my_node"] = build_my_node
```

### 3. Backend — Visualization (`backend/node_viz.py`)

Add forward and/or backward viz functions. Each returns `{ viz?, extras?, insight? }`.

```python
# Forward viz — what to show in forward step-through
def forward_viz_my_node(node_type, module, input_tensor, output, inputs, out_dict):
    result = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    result["insight"] = "What this layer did in plain English"
    # Optional: add extras like weight matrices, kernels, attention maps
    return result

FORWARD_VIZ["ml.layers.my_node"] = forward_viz_my_node

# Backward viz — what to show in backward step-through
def backward_viz_my_node(module, activation, gradient):
    result = {}
    if gradient is not None:
        result["viz"] = viz_vector(gradient.flatten())
    result["insight"] = "What the gradients mean for this layer"
    return result

BACKWARD_VIZ["ml.layers.my_node"] = backward_viz_my_node
```

If you don't add a custom entry, the node gets basic shape-based viz for free via the default fallbacks.

### 4. Backend — Special node types

**Data nodes** — add a loader in `backend/data_loaders.py`:

```python
def load_my_dataset(props: dict) -> dict[str, torch.Tensor]:
    return {"out": images, "labels": labels}

DATA_LOADERS["data.my_dataset"] = load_my_dataset
```

**Loss nodes** — also add to `LOSS_NODES` in `backend/graph_builder.py`:

```python
LOSS_NODES = {"ml.loss.cross_entropy", "ml.loss.my_loss"}
```

**Optimizer nodes** — add to `OPTIMIZER_NODES` and handle in the training loop.

### Summary — files to touch per node type

| Node type | Files to edit |
|-----------|--------------|
| **Layer** (Conv, Linear, etc.) | `src/domain/nodes/<category>/<node>.ts`, folder `index.ts`, `backend/node_builders.py`, `backend/node_viz.py` |
| **Activation** (ReLU, etc.) | `src/domain/nodes/activations/<node>.ts`, folder `index.ts`, `backend/node_builders.py`, `backend/node_viz.py` |
| **Loss** | Same as layer + add to `LOSS_NODES` in `graph_builder.py` |
| **Optimizer** | `src/domain/nodes/optimizers/<node>.ts`, folder `index.ts`, + training loop in `graph_builder.py` |
| **Dataset** | `src/domain/nodes/data/<node>.ts`, folder `index.ts`, `backend/data_loaders.py`, `backend/node_viz.py` |
| **Structural** (Add, Reshape, etc.) | `src/domain/nodes/structural/<node>.ts`, folder `index.ts`, `backend/node_builders.py`, `backend/node_viz.py` |

## Adding a New Data Type

Register it in `src/domain/ml-types.ts`:

```typescript
registry.register({
  id: 'image',
  label: 'Image',
  color: '#f97316',
  compatibleWith: ['tensor'],  // optional: implicit conversions
});
```

The type becomes available for port definitions. Wire colors and connection validation work automatically.

## Adding a New Execution Mode

Register it in `src/domain/ml-modes.ts`:

```typescript
engine.registerMode({
  id: 'profile',
  label: 'Performance Profile',
  propagation: 'manual',
  caching: true,
  executorKey: 'profile',
});
```

Then add `profile` executors to node definitions that support it.

## Adding Validation Rules

All validation lives in `src/core/validation.ts`.

**Forward pass checks** go in `validateForward()`:

```typescript
// Example: check for disconnected input ports
for (const port of ports) {
  if (port.direction === 'input' && !port.optional) {
    const connected = graph.edges.some(
      (e) => e.target.nodeId === nodeId && e.target.portId === port.id,
    );
    if (!connected) {
      errors.push({ nodeId, message: `${def.displayName}: "${port.name}" not connected` });
    }
  }
}
```

**Training-specific checks** go in `validateTraining()`:

```typescript
// Example: check optimizer is connected to loss
for (const optNode of optimizerNodes) {
  const lossConnected = graph.edges.some(
    (e) => e.target.nodeId === optNode.id && e.target.portId === 'loss',
  );
  if (!lossConnected) {
    errors.push({ nodeId: optNode.id, message: 'SGD: loss not connected' });
  }
}
```

When adding new node categories (like loss or optimizer), update the type lists at the top of the file:

```typescript
const LOSS_TYPES = ['ml.loss.cross_entropy', 'ml.loss.mse'];
const OPTIMIZER_TYPES = ['ml.optimizers.sgd', 'ml.optimizers.adam', 'ml.optimizers.adamw'];
const DATA_TYPES = ['data.mnist', 'data.fashion_mnist', 'data.cifar10', 'data.cifar100', 'data.imdb', 'data.ag_news'];
```

## Property Types

Available `type.kind` values for property definitions:

| Kind | UI Widget | Options |
|------|-----------|---------|
| `number` | Number input with +/- buttons | `min`, `max`, `step`, `integer` |
| `string` | Text input | — |
| `boolean` | Checkbox | — |
| `select` | Dropdown | `options: { label, value }[]` |
| `range` | Range slider | `min`, `max` |
| `custom` | Custom React component | `component` (string id) |

## Port Rules

- `dataType` must reference a registered type from `ml-types.ts`
- `allowMultiple: false` means only one edge can connect to this port
- `optional: true` means the node can execute without this port connected
- Port `id` is what executors use to read inputs: `inputs.in`, `inputs.predictions`, etc.
- Output port `id` is what downstream nodes receive: `outputs.out`, `outputs.labels`, etc.

## File Map

```
src/
  core/
    graph.ts         — Layer 1: data structures, topo sort, dirty tracking
    datatypes.ts     — Layer 2: data type registry
    engine.ts        — Layer 3: execution engine
    nodedef.ts       — Layer 4: node definition interfaces, node registry
    validation.ts    — pre-flight validation for forward/training
  domain/
    index.ts         — bootstraps all registrations
    ml-types.ts      — registers tensor, scalar, dataset
    ml-modes.ts      — registers shape, forward, train modes
    nodes/           — one folder per category, one file per node type
  ui/
    useGraph.ts      — bridge between engine and React Flow
    EngineNode.tsx   — generic node renderer
    Toolbar.tsx      — top bar with actions (train, infer, step-through, etc.)
    NodePalette.tsx  — draggable node list
    VizPanel.tsx     — floating per-node visualization (weights, gradients, activations)
    step-through/    — forward + backward step-through panel
      StepThroughPanel.tsx — main panel with Forward/Backward tabs
      StageDetail.tsx      — detail view for selected stage
      ExtraPanels.tsx      — extra viz renderers (kernels, heatmaps, bars, etc.)
      StageTimeline.tsx    — horizontal scrolling timeline
      StageCard.tsx        — individual stage card in timeline
      types.ts             — TypeScript types for step-through data
    dashboard/
      TrainingDashboard.tsx — training progress, charts, system info
    inspector/
      LayerDetail.tsx       — modal with confusion matrix, loss landscape, etc.
      PropertyInspector.tsx — node property editor
      DatasetDetail.tsx     — dataset sample browser

backend/
  main.py            — FastAPI server, endpoints
  graph_builder.py   — graph → PyTorch execution, training loop, inference
  node_builders.py   — per-node-type nn.Module builder functions
  node_viz.py        — per-node-type visualization registry (forward + backward)
  data_loaders.py    — per-dataset loader functions
  step_through.py    — forward step-through orchestration
  backprop_sim.py    — backward step-through + backprop animation
  forward_utils.py   — shared forward-pass node execution
  loss_landscape.py  — loss landscape computation
  runs_store.py      — training run history persistence

model-presets/       — loadable model preset JSON files
```
