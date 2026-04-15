# Custom Blocks (Subgraphs)

Custom blocks let users group several nodes into a single reusable node. A block appears as one node on the outer canvas, but contains a full inner graph that you can enter and edit. This document explains every layer of the implementation.

## Core Concept

A **SubGraph node** (`subgraph.block`) has a `subgraph` field on its `NodeInstance` that holds a complete `Graph` object. Two special **sentinel** node types define the boundary between inside and outside:

| Sentinel | Type string | Role |
|---|---|---|
| **Block Input** | `subgraph.input` | Output ports on this sentinel become **input** ports on the parent block |
| **Block Output** | `subgraph.output` | Input ports on this sentinel become **output** ports on the parent block |

The port direction flips across the boundary: a sentinel's *output* ports face *inward* (delivering data to the inner graph), so they appear as *input* ports on the parent block facing *outward*.

Defined in:
- `src/domain/nodes/subgraph/subgraph.ts` -- the block node definition
- `src/domain/nodes/subgraph/graph-input.ts` -- input sentinel
- `src/domain/nodes/subgraph/graph-output.ts` -- output sentinel

## Creating a Block

When `addNodeToGraph` in `src/ui/useGraph.ts` receives type `"subgraph.block"`, it creates the inner graph and seeds it with default sentinels:

```typescript
if (type === 'subgraph.block') {
  const innerGraph = createGraph(`${id}-inner`, 'Inner Graph');
  const inputNode = createNode('input', 'subgraph.input',
    { x: 0, y: 100 }, { portCount: 1, portNames: 'in' });
  const outputNode = createNode('output', 'subgraph.output',
    { x: 400, y: 100 }, { portCount: 1, portNames: 'out' });
  addNode(innerGraph, inputNode);
  addNode(innerGraph, outputNode);
  node.subgraph = innerGraph;
}
```

The sentinels start with one port each (`in` and `out`). Users can add more ports by changing the `portCount` / `portNames` properties on the sentinels. Port names are comma-separated strings parsed by `getPorts()`.

## Port Resolution

The subgraph block's own `NodeDefinition.getPorts()` returns an empty array -- it is intentionally a no-op. Instead, `getNodePorts()` in `src/core/ports.ts` detects `node.subgraph` and delegates to `getSubgraphPorts()`:

```typescript
export function getNodePorts(node: NodeInstance, registry: NodeRegistry): PortDefinition[] {
  if (node.subgraph) {
    return getSubgraphPorts(node.subgraph, registry);
  }
  const def = registry.get(node.type);
  if (!def) return [];
  return def.getPorts(node.properties);
}
```

`getSubgraphPorts()` iterates every node in the inner graph. For each sentinel it finds:

- `subgraph.input` nodes: takes each of their ports and re-emits them with `direction: 'input'` (the flip).
- `subgraph.output` nodes: takes each of their ports and re-emits them with `direction: 'output'`.

This means the parent block's visible ports are always derived live from whatever sentinels exist inside. Adding a second `subgraph.input` sentinel or changing `portCount` on an existing one immediately changes the parent's ports.

## Navigation

Users double-click a block node to enter its inner graph. The navigation state lives in `useGraph.ts`:

```typescript
interface NavEntry {
  graphId: string;
  label: string;
  nodeId: string; // the subgraph node's id in the parent graph
}

const [navStack, setNavStack] = useState<NavEntry[]>([]);
```

### `getCurrentGraph()`

Every graph operation -- add node, remove node, connect, validate, sync to React Flow -- calls `getCurrentGraph()` to get the graph the user is currently viewing:

```typescript
const getCurrentGraph = useCallback((): Graph => {
  let g = graphRef.current;              // start at root
  for (const entry of navStack) {         // walk into each nested level
    const node = g.nodes.get(entry.nodeId);
    if (node?.subgraph) {
      g = node.subgraph;
    } else {
      break;
    }
  }
  return g;
}, [navStack]);
```

If `navStack` is empty, the root graph is returned. If the user has entered a block, the stack has one entry; entering a block-within-a-block adds another entry.

### `enterSubgraph(nodeId)`

Called on double-click. Pushes a new `NavEntry` onto `navStack`:

```typescript
const enterSubgraph = useCallback((nodeId: string) => {
  const currentGraph = getCurrentGraph();
  const node = currentGraph.nodes.get(nodeId);
  if (!node || node.type !== 'subgraph.block' || !node.subgraph) return;

  setNavStack((prev) => [
    ...prev,
    { graphId: node.subgraph!.id, label: node.properties.blockName || 'Block', nodeId },
  ]);
}, [getCurrentGraph]);
```

### `navigateTo(depth)`

Called from the breadcrumb UI. Truncates `navStack` to `depth` entries:

```typescript
const navigateTo = useCallback((depth: number) => {
  setNavStack((prev) => prev.slice(0, depth));
}, []);
```

Passing `0` returns to the root graph.

### Re-sync on navigation

A `useEffect` watches `navStack` and calls `syncToRF()` whenever it changes, so React Flow immediately renders the inner graph's nodes and edges:

```typescript
useEffect(() => { syncToRF(); }, [navStack]);
```

## How Operations Target the Right Graph

Because `addNodeToGraph`, `removeNodeFromGraph`, `connectNodes`, `updateProperty`, `onNodesChange`, `onEdgesChange`, and `isValidConnection` all call `getCurrentGraph()`, they naturally operate on whatever graph the user is viewing. No special subgraph logic is needed in these functions -- they see a `Graph` and work on it.

For example, when the user is inside a block and connects two nodes, `connectNodes` adds the edge to the inner graph (via `getCurrentGraph()`), then calls `markDirty` on that inner graph, then runs shape inference on the root graph (which recurses into the inner graph).

## Engine Recursion (Shape Inference)

`src/core/engine.ts` -- the `execute()` method -- handles subgraph nodes with an explicit recursive branch. When it encounters a node with `node.subgraph`, it does not look up an executor. Instead:

### Step 1: Inject inputs into the input sentinel

The engine iterates the inner graph's nodes, finds all `subgraph.input` sentinels, and writes the parent node's gathered inputs directly onto the sentinel's `lastResult`:

```typescript
for (const [, innerNode] of node.subgraph.nodes) {
  if (innerNode.type === 'subgraph.input') {
    innerNode.lastResult = {
      outputs: inputs,     // the data flowing into the parent block
      metadata: { ... },
    };
    innerNode.dirty = false;
  } else {
    innerNode.dirty = true;  // force re-execution of all other inner nodes
  }
}
```

### Step 2: Recurse

The engine calls itself on the inner graph:

```typescript
await this.execute(node.subgraph, modeId, lookupExecutor);
```

This runs the full topological-sort-and-execute loop on the inner graph. Each inner node runs its own executor (Conv2d does shape math, ReLU passes through, etc.). This recurse is unbounded -- blocks can contain blocks.

### Step 3: Read outputs from the output sentinel

After the inner execution completes, the engine reads `lastResult.outputs` from all `subgraph.output` sentinels and merges them:

```typescript
const outputs: Record<string, any> = {};
for (const [, innerNode] of node.subgraph.nodes) {
  if (innerNode.type === 'subgraph.output' && innerNode.lastResult?.outputs) {
    Object.assign(outputs, innerNode.lastResult.outputs);
  }
}
```

### Step 4: Aggregate metadata

The engine sums `paramCount` from all inner nodes and writes it as the block's total:

```typescript
let totalParams = 0;
const paramParts: string[] = [];
for (const [, innerNode] of node.subgraph.nodes) {
  const pc = innerNode.lastResult?.metadata?.paramCount;
  if (pc) {
    totalParams += pc;
    paramParts.push(`${shortName}: ${pc.toLocaleString()}`);
  }
}
```

The block's `lastResult` gets the merged outputs, total param count, and breakdown string.

## Backend Execution

`backend/graph_builder.py` mirrors the frontend recursion with real PyTorch tensors.

### `SubGraphModule`

A `SubGraphModule` (extends `nn.Module`) wraps the inner graph's modules:

```python
class SubGraphModule(nn.Module):
    def __init__(self, inner_modules, inner_nodes, inner_edges, inner_order):
        super().__init__()
        self.inner_modules = nn.ModuleDict()  # registered so parameters are visible
        for nid, mod in inner_modules.items():
            safe_key = nid.replace('.', '_').replace('-', '_')
            self.inner_modules[safe_key] = mod
```

Using `nn.ModuleDict` ensures that PyTorch's parameter discovery (used by optimizers) sees all parameters inside the block.

### `SubGraphModule.forward(**inputs)`

The forward method walks nodes in topological order:

1. **Sentinel input**: stores the parent's `**inputs` dict as the sentinel's results.
2. **Sentinel output**: gathers its upstream inputs and returns them.
3. **Regular nodes**: looks up the module, calls it, stores the output.

```python
def forward(self, **inputs):
    results = {}
    for node_id in self.inner_order:
        node_type = self.inner_nodes[node_id]["type"]
        if node_type == SENTINEL_INPUT:
            results[node_id] = inputs
            continue
        if node_type == SENTINEL_OUTPUT:
            results[node_id] = gather_inputs(node_id, self.inner_edges, results)
            continue
        # ... run module ...
    # Return output sentinel's results
    for node_id in self.inner_order:
        if self.inner_nodes[node_id]["type"] == SENTINEL_OUTPUT:
            return results.get(node_id, {})
```

### `build_subgraph_module(subgraph_data, parent_input_shapes)`

Builds the `SubGraphModule` from serialized data. It does a shape-propagation dry run first (creating dummy tensors to determine output shapes), builds each inner module with the correct dimensions, then wraps them all in a `SubGraphModule`.

### Usage in `build_and_run_graph`

When the main graph walk encounters a `subgraph.block` node:

```python
if node_type == SUBGRAPH_TYPE:
    sg_module = build_subgraph_module(subgraph_data, input_shapes)
    modules[node_id] = sg_module
    sg_outputs = sg_module(**inputs)
```

The `modules[node_id] = sg_module` line means the optimizer sees the block's parameters during training. The training loop calls `sg_module(**inputs)` identically to any other module.

## Saving and Loading Blocks

### Save

`saveBlock(nodeId)` in `useGraph.ts` serializes the inner graph and POSTs it to the backend:

```typescript
const blockData = {
  name: node.properties.blockName || 'Custom Block',
  description: `Custom block with ${node.subgraph.nodes.size} nodes`,
  subgraph: serializeGraphData(node.subgraph),
};
await fetch('http://localhost:8000/blocks/save', { method: 'POST', body: JSON.stringify(blockData) });
```

The backend (`POST /blocks/save` in `backend/main.py`) writes the JSON to `backend/blocks/<name>.json`.

### Load

`addBlockFromTemplate(filename, position)` fetches the block JSON from the backend, deserializes the inner graph, and creates a new subgraph node:

```typescript
const node = createNode(id, 'subgraph.block', position, { blockName: block.name });
node.subgraph = deserializeGraphData(block.subgraph);
addNode(currentGraph, node);
```

### Storage locations

| Directory | Purpose | Mutability |
|---|---|---|
| `backend/presets/` | Shipped block templates (e.g., Transformer Encoder Layer) | Read-only |
| `backend/blocks/` | User-saved blocks | Read/write/delete |

The `GET /blocks` endpoint lists both, tagging presets with `"preset": true`. The `DELETE /blocks/{filename}` endpoint refuses to delete presets.

## Serialization

`serializeGraphData` in `useGraph.ts` is recursive. For each node, if it has a `subgraph`, the function calls itself on the inner graph:

```typescript
function serializeGraphData(graph: Graph): SerializedGraphData {
  return {
    id: graph.id,
    name: graph.name,
    nodes: Array.from(graph.nodes.values()).map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      properties: n.properties,
      ...(n.subgraph ? { subgraph: serializeGraphData(n.subgraph) } : {}),
    })),
    edges: graph.edges.map((e) => ({ id: e.id, source: e.source, target: e.target })),
  };
}
```

`deserializeGraphData` mirrors this, recursively restoring `node.subgraph` for any serialized node that has one. This means graphs with blocks-inside-blocks serialize and deserialize correctly at any nesting depth.

## End-to-End Trace

Scenario: MNIST -> Custom Block (contains Conv2d -> ReLU) -> Flatten, with batch size 1.

### Outer graph structure

```
[MNIST] --out--> [Custom Block] --out--> [Flatten]
```

### Inner graph structure (inside the Custom Block)

```
[Block Input] --in--> [Conv2d(32, k=3)] --out--> [ReLU] --out--> [Block Output]
```

### Shape inference walk

1. **Topological sort** of the outer graph: `[MNIST, Custom Block, Flatten]`.

2. **MNIST** runs its shape executor. Result: `outputs.out = [1, 1, 28, 28]`, `outputs.labels = [1]`.

3. **Custom Block** is reached. The engine sees `node.subgraph` and enters the recursive path:

   a. **Inject inputs**: The engine gathers the Custom Block's inputs from upstream edges. MNIST's `out` port connects to the block's `in` port, so `inputs = { in: [1, 1, 28, 28] }`. This dict is written to the Block Input sentinel's `lastResult.outputs`.

   b. **Recurse**: `engine.execute(innerGraph, 'shape', lookupExecutor)` is called.

   c. **Inner topological sort**: `[Block Input, Conv2d, ReLU, Block Output]`.

   d. **Block Input**: already has `lastResult` (injected in step a), marked clean. Skipped by caching. Its outputs are `{ in: [1, 1, 28, 28] }`.

   e. **Conv2d(32, k=3, s=1, p=0)**: gathers input from Block Input's `in` port. Runs the shape formula:
      - `outH = floor((28 + 2*0 - 3) / 1) + 1 = 26`
      - `outW = floor((28 + 2*0 - 3) / 1) + 1 = 26`
      - Output: `[1, 32, 26, 26]`
      - Params: `32 * 1 * 3 * 3 + 32 = 320`

   f. **ReLU**: passes shape through unchanged. Output: `[1, 32, 26, 26]`.

   g. **Block Output**: gathers input from ReLU's `out` port. Its executor returns `outputs = { out: [1, 32, 26, 26] }`.

   h. **Read output sentinels**: the engine reads Block Output's `lastResult.outputs` -> `{ out: [1, 32, 26, 26] }`.

   i. **Aggregate params**: Conv2d had `paramCount: 320`, ReLU had none. Block total: 320. Breakdown: `"conv2d: 320 = 320"`.

   j. **Store on parent**: the Custom Block node's `lastResult` becomes `{ outputs: { out: [1, 32, 26, 26] }, metadata: { outputShape: [1, 32, 26, 26], paramCount: 320, ... } }`.

4. **Flatten**: gathers input `[1, 32, 26, 26]` from the Custom Block's `out` port. Computes `flat = 32 * 26 * 26 = 21632`. Output: `[1, 21632]`.

### Data flow summary

```
MNIST          ->  [1, 1, 28, 28]
  |
Custom Block   ->  [1, 32, 26, 26]    (320 params)
  |  inner:
  |    Block Input   [1, 1, 28, 28]
  |    Conv2d        [1, 32, 26, 26]
  |    ReLU          [1, 32, 26, 26]
  |    Block Output  [1, 32, 26, 26]
  |
Flatten        ->  [1, 21632]
```
