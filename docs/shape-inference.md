# Shape Inference

Shape inference is the "shape" execution mode. It runs entirely in TypeScript with no backend, computes output tensor dimensions for every node, and updates the canvas in real time as the user edits the graph. This document traces the full path from user action to displayed shape.

## The "shape" Execution Mode

Registered in `src/domain/ml-modes.ts`:

```typescript
engine.registerMode({
  id: 'shape',
  label: 'Shape Inference',
  propagation: 'eager',    // auto-run whenever nodes are dirty
  caching: true,            // skip nodes whose inputs haven't changed
  executorKey: 'shape',     // look up the "shape" key in each node's executors map
});
```

Key properties:

- **Eager propagation**: shape inference runs automatically after every graph mutation. The user never clicks "run" -- it happens on every add, remove, connect, disconnect, or property change.
- **Caching**: nodes that are not dirty are skipped. Only the changed node and its downstream dependents re-execute.
- **TypeScript math**: shape executors do arithmetic on dimension arrays (like `[1, 64, 26, 26]`). No tensors are allocated, no backend is called.

## Engine Flow

`src/core/engine.ts` -- `ExecutionEngine.execute()`:

### 1. Topological sort

```typescript
const order = topologicalSort(graph);
```

Uses Kahn's algorithm from `src/core/graph.ts`. Returns node IDs in dependency order -- upstream nodes before downstream nodes. Throws if the graph has a cycle.

### 2. For each node in order

```typescript
for (const nodeId of order) {
  const node = graph.nodes.get(nodeId)!;
```

### 3. Skip clean nodes (caching)

```typescript
if (mode.caching && !node.dirty) continue;
```

If the node's `dirty` flag is `false`, its inputs haven't changed since the last run. The engine skips it and moves on. The node's `lastResult` from the previous run is still valid and still available to downstream nodes.

### 4. Gather inputs from upstream

```typescript
const inputs: Record<string, any> = {};
for (const edge of graph.edges) {
  if (edge.target.nodeId !== nodeId) continue;
  const sourceNode = graph.nodes.get(edge.source.nodeId);
  if (sourceNode?.lastResult?.outputs) {
    inputs[edge.target.portId] = sourceNode.lastResult.outputs[edge.source.portId];
  }
}
```

For each incoming edge, the engine reads the upstream node's `lastResult.outputs[portId]` and maps it to the target port id. For shape mode, these values are dimension arrays like `[1, 1, 28, 28]`.

### 5. Handle subgraph nodes (recursive)

If `node.subgraph` exists, the engine injects inputs into the inner graph's input sentinels, recursively calls `execute()` on the inner graph, then reads outputs from the output sentinels. See [custom-blocks.md](./custom-blocks.md) for the full details.

### 6. Look up and run the executor

```typescript
const executor = lookupExecutor(node.type, mode.executorKey);
if (!executor) continue;

const context: ExecutionContext = {
  inputs,
  properties: node.properties,
  state: node.state,
  mode: modeId,
};

const result = await executor.execute(context);
```

The `lookupExecutor` function is `NodeRegistry.getExecutor()` -- it finds the `NodeDefinition` for the node type and returns `definition.executors["shape"]`. If the node has no shape executor (e.g., optimizer nodes), it is skipped.

### 7. Store the result

```typescript
node.lastResult = result;
if (result.state !== undefined) {
  node.state = result.state;
}
node.dirty = false;
```

The result is written to `NodeInstance.lastResult`, which the UI reads for display. The `dirty` flag is cleared.

## Dirty Tracking

Dirty tracking ensures the engine only re-runs what has changed. It works through two mechanisms:

### `setProperty()` in `src/core/graph.ts`

When a property changes, the node and all its downstream dependents are marked dirty:

```typescript
export function setProperty(graph: Graph, nodeId: string, key: string, value: any): void {
  const node = graph.nodes.get(nodeId);
  node.properties[key] = value;
  markDirty(graph, nodeId);
}
```

### `markDirty()` -- BFS downstream

```typescript
export function markDirty(graph: Graph, nodeId: string): void {
  const node = graph.nodes.get(nodeId);
  node.dirty = true;

  const visited = new Set<string>([nodeId]);
  const queue = [nodeId];

  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const edge of graph.edges) {
      if (edge.source.nodeId === current && !visited.has(edge.target.nodeId)) {
        visited.add(edge.target.nodeId);
        queue.push(edge.target.nodeId);
        graph.nodes.get(edge.target.nodeId)!.dirty = true;
      }
    }
  }
}
```

This is a BFS walk following edges forward. If you change Conv2d's kernel size, Conv2d and every node reachable downstream (ReLU, Flatten, Linear, etc.) are marked dirty.

### Connecting and disconnecting edges

In `useGraph.ts`, both `connectNodes` and `onEdgesChange` (for removals) call `markDirty` on the target node:

```typescript
// connectNodes:
addGraphEdge(currentGraph, edge);
markDirty(currentGraph, connection.target);

// onEdgesChange (removal):
const edge = g.edges.find((e) => e.id === change.id);
if (edge) {
  markDirty(g, edge.target.nodeId);
}
```

### Auto-run after mutations

Every mutation function in `useGraph.ts` ends with `await runShape()`:

```typescript
const runShape = useCallback(async () => {
  await domain.engine.execute(
    graphRef.current,
    'shape',
    (nodeType, executorKey) => domain.nodeRegistry.getExecutor(nodeType, executorKey),
  );
  syncToRF();
}, [domain, syncToRF]);
```

Note: `runShape` always executes on `graphRef.current` (the root graph), not `getCurrentGraph()`. The engine's recursive subgraph handling ensures inner graphs are re-executed when needed.

## How Metadata Flows to the UI

Each executor returns an `ExecutionResult`:

```typescript
interface ExecutionResult {
  outputs: Record<string, any>;   // port data for downstream nodes
  state?: any;                     // persistent state (weights, etc.)
  metadata?: Record<string, any>; // display data for the UI
}
```

The engine stores this on `NodeInstance.lastResult`. The UI component (`src/ui/EngineNode.tsx`) reads `lastResult.metadata` to render:

- **`shapes`**: an array of `{ label, value }` pairs displayed as labeled rows on the node. Example: `[{ label: "Output", value: [1, 64, 26, 26] }, { label: "Weights", value: [64, 1, 3, 3] }]`.
- **`outputShape`**: fallback if `shapes` is not present. Shows a single shape.
- **`paramCount`**: displayed as a badge, e.g., "640 params".
- **`paramBreakdown`**: shown in the inspector panel, e.g., `"weights: 64x1x3x3 = 576 + bias: 64 = 640"`.
- **`error`**: a red error box displayed on the node.

`syncToRF()` converts the graph to React Flow format, spreading `{ ...node }` to create a new object reference so React detects the change and re-renders.

## Example Executors

### MNIST (data source, no inputs)

`src/domain/nodes/data/mnist.ts`:

```typescript
execute: async ({ properties }) => {
  const B = properties.batchSize;
  return {
    outputs: {
      out: [B, 1, 28, 28],
      labels: [B],
    },
    metadata: {
      shapes: [
        { label: 'Images', value: [B, 1, 28, 28] },
        { label: 'Labels', value: [B] },
        { label: 'Classes', value: '10' },
        // ...
      ],
    },
  };
}
```

No inputs needed -- MNIST is always the starting point. It produces a shape based solely on the `batchSize` property.

### Conv2d (parametric layer)

`src/domain/nodes/layers/conv2d.ts`:

```typescript
execute: async ({ inputs, properties }) => {
  const input = inputs.in;
  if (!input) return { outputs: {} };

  const [B, C, H, W] = input;
  const { outChannels: OC, kernelSize: K, stride: S, padding: P } = properties;
  const outH = Math.floor((H + 2 * P - K) / S) + 1;
  const outW = Math.floor((W + 2 * P - K) / S) + 1;

  if (outH <= 0 || outW <= 0) {
    return {
      outputs: {},
      metadata: { error: `Invalid output: ${outH}x${outW} (kernel ${K} too large for ${H}x${W} input)` },
    };
  }

  return {
    outputs: { out: [B, OC, outH, outW] },
    metadata: {
      paramCount: OC * C * K * K + OC,
      paramBreakdown: `weights: ${OC}x${C}x${K}x${K} = ${OC * C * K * K}  +  bias: ${OC}  =  ${OC * C * K * K + OC}`,
      outputShape: [B, OC, outH, outW],
      shapes: [
        { label: 'Output', value: [B, OC, outH, outW] },
        { label: 'Weights', value: [OC, C, K, K] },
        { label: 'Bias', value: [OC] },
      ],
    },
  };
}
```

This is the standard convolution formula: `output_size = floor((input + 2*padding - kernel) / stride) + 1`. The executor also validates the result and returns an error if dimensions go non-positive.

### ReLU (passthrough)

`src/domain/nodes/activations/relu.ts`:

```typescript
execute: async ({ inputs }) => {
  const input = inputs.in;
  if (!input) return { outputs: {} };
  return {
    outputs: { out: input },
    metadata: { outputShape: input, shapes: [{ label: 'Output', value: input }] },
  };
}
```

Activation functions do not change tensor dimensions. The shape passes straight through.

### Flatten

`src/domain/nodes/layers/flatten.ts`:

```typescript
execute: async ({ inputs }) => {
  const input = inputs.in;
  if (!input) return { outputs: {} };
  const batch = input[0];
  const flat = input.slice(1).reduce((a, b) => a * b, 1);
  return {
    outputs: { out: [batch, flat] },
    metadata: { outputShape: [batch, flat], shapes: [{ label: 'Output', value: [batch, flat] }] },
  };
}
```

Multiplies all dimensions after batch into a single number. `[1, 32, 26, 26]` becomes `[1, 32*26*26]` = `[1, 21632]`.

### Linear

`src/domain/nodes/layers/linear.ts`:

```typescript
execute: async ({ inputs, properties }) => {
  const input = inputs.in;
  if (!input) return { outputs: {} };
  const inFeatures = input[input.length - 1];
  const outFeatures = properties.outFeatures;
  const outShape = [...input.slice(0, -1), outFeatures];
  return {
    outputs: { out: outShape },
    metadata: {
      paramCount: inFeatures * outFeatures + outFeatures,
      // ...
    },
  };
}
```

Replaces the last dimension with `outFeatures`. The `inFeatures` dimension is read from the input shape, so `Linear` auto-adapts to whatever feeds into it.

## End-to-End Trace

Scenario: MNIST -> Conv2d(32, k=3) -> ReLU -> Flatten -> Linear(10), with batch size 1.

### Initial state

The user drops nodes and connects them one by one. After each connection, `runShape()` fires.

### Topological order

`[MNIST, Conv2d, ReLU, Flatten, Linear]`

### Step-by-step execution

**MNIST** (no inputs, dirty=true):
- Runs its shape executor
- outputs: `{ out: [1, 1, 28, 28], labels: [1] }`
- dirty -> false

**Conv2d** (outChannels=32, kernelSize=3, stride=1, padding=0):
- Gathers inputs: `{ in: [1, 1, 28, 28] }` (from MNIST's `out` port via the edge)
- Destructures: B=1, C=1, H=28, W=28
- Computes: outH = floor((28 + 0 - 3) / 1) + 1 = **26**, outW = **26**
- Computes params: 32 * 1 * 3 * 3 + 32 = **320**
- outputs: `{ out: [1, 32, 26, 26] }`
- metadata: `{ paramCount: 320, paramBreakdown: "weights: 32x1x3x3 = 288 + bias: 32 = 320", shapes: [...] }`

**ReLU**:
- Gathers inputs: `{ in: [1, 32, 26, 26] }` (from Conv2d's `out`)
- Passes through unchanged
- outputs: `{ out: [1, 32, 26, 26] }`

**Flatten**:
- Gathers inputs: `{ in: [1, 32, 26, 26] }` (from ReLU's `out`)
- Computes: batch=1, flat = 32 * 26 * 26 = **21632**
- outputs: `{ out: [1, 21632] }`

**Linear(10)**:
- Gathers inputs: `{ in: [1, 21632] }` (from Flatten's `out`)
- Reads: inFeatures = 21632, outFeatures = 10
- Computes: outShape = [1, 10]
- Computes params: 21632 * 10 + 10 = **216,330**
- outputs: `{ out: [1, 10] }`
- metadata: `{ paramCount: 216330, paramBreakdown: "weights: 21632x10 = 216320 + bias: 10 = 216330", shapes: [...] }`

### Shape flow summary

```
MNIST       [1, 1, 28, 28]                     0 params
    |
Conv2d      [1, 32, 26, 26]                  320 params
    |
ReLU        [1, 32, 26, 26]                    0 params
    |
Flatten     [1, 21632]                         0 params
    |
Linear      [1, 10]                       216,330 params
                                          --------
                                Total:    216,650 params
```

### What the user sees

Each node on the canvas displays its output shape and parameter count in real time. The moment the user changes Conv2d's kernel size from 3 to 5:

1. `setProperty` writes `kernelSize = 5` and calls `markDirty` on Conv2d.
2. `markDirty` BFS marks Conv2d, ReLU, Flatten, and Linear as dirty.
3. `runShape()` calls `engine.execute(graph, 'shape', ...)`.
4. MNIST is clean (not dirty) -- skipped.
5. Conv2d re-runs: outH = floor((28 + 0 - 5) / 1) + 1 = **24**. New output: `[1, 32, 24, 24]`. Params: 32 * 1 * 5 * 5 + 32 = **832**.
6. ReLU re-runs: passes `[1, 32, 24, 24]` through.
7. Flatten re-runs: 32 * 24 * 24 = **18432**. Output: `[1, 18432]`.
8. Linear re-runs: 18432 * 10 + 10 = **184,330** params. Output: `[1, 10]`.
9. `syncToRF()` pushes the new node data to React Flow. All four nodes update instantly on the canvas.

### Error case

If the user sets kernel size to 30 (larger than the 28x28 input):

1. Conv2d computes: outH = floor((28 + 0 - 30) / 1) + 1 = **-1**.
2. The executor detects `outH <= 0` and returns `{ outputs: {}, metadata: { error: "Invalid output: -1x-1 (kernel 30 too large for 28x28 input)" } }`.
3. Since Conv2d's `outputs` is empty, downstream nodes receive no input. Their executors return `{ outputs: {} }` (the `if (!input)` guard).
4. The UI shows a red error on the Conv2d node and clears shapes from downstream nodes.
