# Contributing — How to Add New Functionality

## Adding a New Node Type

### 1. Frontend (shape inference)

Create a file in the appropriate category folder under `src/domain/nodes/`:

```
src/domain/nodes/
  data/          — dataset nodes (MNIST, CIFAR-10)
  layers/        — neural network layers (Conv2d, Linear, Flatten)
  activations/   — activation functions (ReLU, Sigmoid, Softmax)
  loss/          — loss functions (CrossEntropyLoss, MSELoss)
  optimizers/    — optimizers (SGD, Adam)
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

### 2. Backend (forward pass + training)

Add a builder function in `backend/node_builders.py`:

```python
def build_my_node(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    return nn.MyModule(
        some_param=props["someParam"],
        # Derive input-dependent args from in_shape
    )

NODE_BUILDERS["ml.layers.my_node"] = build_my_node
```

For data nodes, add a loader in `backend/data_loaders.py`:

```python
def load_my_dataset(props: dict) -> dict[str, torch.Tensor]:
    # Return a dict of port_id → tensor
    return {"out": images, "labels": labels}

DATA_LOADERS["data.my_dataset"] = load_my_dataset
```

For loss nodes, also add the type to `LOSS_NODES` in `backend/graph_builder.py`:

```python
LOSS_NODES = {"ml.loss.cross_entropy", "ml.loss.my_loss"}
```

For optimizer nodes, add to `OPTIMIZER_NODES` and handle in the training loop.

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
const OPTIMIZER_TYPES = ['ml.optimizers.sgd', 'ml.optimizers.adam'];
const DATA_TYPES = ['data.mnist', 'data.cifar10'];
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
    PropertyInspector.tsx
    NodePalette.tsx
    Toolbar.tsx

backend/
  main.py            — FastAPI server, endpoints
  graph_builder.py   — graph → PyTorch execution
  node_builders.py   — per-node-type module builders
  data_loaders.py    — per-dataset loader functions
```
