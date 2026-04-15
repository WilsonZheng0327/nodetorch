# Multi-Output Nodes (LSTM / GRU)

Most nodes in NodeTorch have a single input and a single output -- data flows in through `in` and out through `out`. Recurrent nodes break this pattern. A PyTorch LSTM returns a tuple `(output, (hidden, cell))` -- three separate tensors, each with a different shape and meaning. A GRU returns `(output, hidden)` -- two tensors. This document traces how the codebase handles multi-output nodes end-to-end, from port definitions through shape inference, backend execution, edge routing, training, and inference.

---

## 1. The problem: tuples don't fit the single-output model

In vanilla PyTorch:

```python
output, (hidden, cell) = lstm(x)
```

`output` has shape `[B, seq_len, hidden_size]`, `hidden` has shape `[num_layers, B, hidden_size]`, and `cell` has the same shape as `hidden`. These are three different tensors that downstream nodes might each need independently. A downstream Linear layer might want `output`, while a second LSTM in a stacked architecture might want both `hidden` and `cell` as initial states.

A single-output `"out"` port cannot represent this. The solution in NodeTorch is to give multi-output nodes **multiple named output ports**, one per tensor, and then thread that naming convention through every layer of the architecture.

---

## 2. Concrete example: LSTM with input [32, 10, 64]

Throughout this document, we follow a single concrete example:

- **LSTM node** with properties: `hiddenSize=128`, `numLayers=1`, `bidirectional=false`
- **Input tensor**: shape `[32, 10, 64]` (batch=32, seq_len=10, input_size=64)

Expected outputs:
| Port | Shape | Meaning |
|------|-------|---------|
| `out` | `[32, 10, 128]` | Full output sequence -- one 128-dim vector per timestep |
| `hidden` | `[1, 32, 128]` | Final hidden state (num_layers * directions, batch, hidden) |
| `cell` | `[1, 32, 128]` | Final cell state (same shape as hidden) |

---

## 3. Frontend: port definitions

### `src/domain/nodes/layers/lstm.ts`, lines 24-29

```typescript
getPorts: () => [
  { id: 'in',     name: 'Input',  direction: 'input',  dataType: 'tensor', allowMultiple: false, optional: false },
  { id: 'out',    name: 'Output', direction: 'output', dataType: 'tensor', allowMultiple: true,  optional: false },
  { id: 'hidden', name: 'Hidden', direction: 'output', dataType: 'tensor', allowMultiple: true,  optional: false },
  { id: 'cell',   name: 'Cell',   direction: 'output', dataType: 'tensor', allowMultiple: true,  optional: false },
],
```

Key details:
- There is **one input port** (`in`) and **three output ports** (`out`, `hidden`, `cell`).
- All output ports have `allowMultiple: true`, meaning multiple downstream nodes can connect to the same output port.
- The port `id` values (`"out"`, `"hidden"`, `"cell"`) are the keys used everywhere -- in the edge data structure, in executor results, in the backend's result dictionaries. They must match exactly across the entire stack.

For comparison, GRU (`src/domain/nodes/layers/gru.ts`, lines 23-27) defines only two output ports (`out` and `hidden`) since GRU has no cell state.

### How ports become React Flow handles

`src/ui/EngineNode.tsx`, lines 155-176, renders each port as a React Flow `Handle`. The output ports are iterated at lines 167-176:

```typescript
{outputPorts.map((port, i) => (
  <div key={port.id} className="port port-output" style={{ top: `${((i + 1) / (outputPorts.length + 1)) * 100}%` }}>
    <span className="port-label port-label-right">{port.name}</span>
    <RF.Handle id={port.id} type="source" position={RF.Position.Right} />
  </div>
))}
```

The `Handle`'s `id` is set to `port.id` (e.g., `"hidden"`). When a user drags a wire from this handle, React Flow records the `sourceHandle` as `"hidden"`. This is the link between the visual wire and the data model.

---

## 4. Frontend: shape executor

### `src/domain/nodes/layers/lstm.ts`, lines 31-68

The shape executor runs in TypeScript with no backend. It computes all three output shapes using pure arithmetic.

```typescript
execute: async ({ inputs, properties }) => {
  const input = inputs.in;
  if (!input) return { outputs: {} };

  if (input.length !== 3) {
    return { outputs: {}, metadata: { error: `Input should be [B, seq_len, input_size], got [${input}]` } };
  }

  const [B, seqLen, inputSize] = input;
  const { hiddenSize: H, numLayers: L, bidirectional } = properties;
  const D = bidirectional ? 2 : 1;

  const outShape = [B, seqLen, H * D];
  const hiddenShape = [L * D, B, H];
  const cellShape = [L * D, B, H];
  // ...
  return {
    outputs: { out: outShape, hidden: hiddenShape, cell: cellShape },
    metadata: { ... },
  };
},
```

**For our example** (input `[32, 10, 64]`, H=128, L=1, D=1):
- `outShape` = `[32, 10, 128]`
- `hiddenShape` = `[1, 32, 128]`
- `cellShape` = `[1, 32, 128]`

The return value's `outputs` map has **three keys** matching the three port ids. This is how the engine knows what data to pass downstream through each port.

### How shape results propagate to downstream nodes

The execution engine (`src/core/engine.ts`, lines 112-119) gathers inputs for each node by scanning edges:

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

This is the critical routing logic. For an edge from LSTM's `hidden` port to a Linear node's `in` port:
- `edge.source.portId` = `"hidden"`
- `edge.target.portId` = `"in"`
- `sourceNode.lastResult.outputs["hidden"]` = `[1, 32, 128]` (the shape array)
- So `inputs["in"]` on the downstream Linear gets `[1, 32, 128]`

The same code handles single-output nodes transparently -- when a node only has `outputs: { out: [32, 10, 128] }`, the edge's `source.portId` is `"out"` and it reads `outputs["out"]`.

---

## 5. Edge data structure

Edges are defined in `src/core/graph.ts`, lines 57-63:

```typescript
export interface Edge {
  id: string;
  source: { nodeId: string; portId: string };
  target: { nodeId: string; portId: string };
}
```

When a user drags a wire from an LSTM's "Hidden" handle to a Linear's "Input" handle, `connectNodes` in `src/ui/useGraph.ts` (lines 381-403) creates:

```typescript
{
  id: "e-1713168000001",
  source: { nodeId: "ml.layers.lstm-1713168000000", portId: "hidden" },
  target: { nodeId: "ml.layers.linear-1713168000002", portId: "in" }
}
```

The `portId` on the source side selects **which** of the LSTM's three outputs flows through this edge. Different edges from the same LSTM node can carry different outputs. For instance, a second edge might route `portId: "cell"` to a different downstream node.

### Connection validation

`isValidConnection` in `src/ui/useGraph.ts` (lines 288-333) validates connections before they are created. It checks:

1. No self-connections (line 296)
2. Both ports exist on their respective nodes (lines 306-314)
3. Data type compatibility via Layer 2 (lines 317-319) -- all LSTM output ports are `dataType: "tensor"`, so they are compatible with any input port that accepts `"tensor"`
4. Single-connection input ports are not already occupied (lines 322-329)

This validation works identically for multi-output nodes -- each output port is validated independently.

---

## 6. Backend: LSTMWrapper

### `backend/node_builders.py`, lines 84-93

PyTorch's `nn.LSTM.forward()` returns a tuple. NodeTorch's backend needs dict outputs keyed by port id. The `LSTMWrapper` class bridges this gap:

```python
class LSTMWrapper(nn.Module):
    """Wraps nn.LSTM to return a dict of named outputs."""
    def __init__(self, lstm: nn.LSTM):
        super().__init__()
        self.lstm = lstm

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, (hidden, cell) = self.lstm(x)
        return {"out": output, "hidden": hidden, "cell": cell}
```

The dict keys (`"out"`, `"hidden"`, `"cell"`) **must match** the frontend port ids exactly. If they diverge, `gather_inputs` will fail to find the right tensor for downstream nodes.

`build_lstm` (`backend/node_builders.py`, lines 106-117) constructs the wrapper:

```python
def build_lstm(props: dict, input_shapes: dict) -> nn.Module:
    in_shape = input_shapes.get("in")
    input_size = in_shape[-1] if in_shape else 1
    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=props["hiddenSize"],
        num_layers=props.get("numLayers", 1),
        batch_first=True,
        bidirectional=props.get("bidirectional", False),
        dropout=props.get("dropout", 0) if props.get("numLayers", 1) > 1 else 0,
    )
    return LSTMWrapper(lstm)
```

For our example: `input_size=64`, `hidden_size=128`, `num_layers=1`, `batch_first=True`, `bidirectional=False`.

GRU has an analogous wrapper (`GRUWrapper`, lines 95-103) that returns `{"out": output, "hidden": hidden}` -- two keys instead of three, matching GRU's two output ports.

---

## 7. Backend: forward pass -- detecting and storing dict outputs

### `backend/graph_builder.py`, `build_and_run_graph()`, lines 354-411

When the forward pass encounters a layer node, it runs the module and checks the return type:

```python
raw_output = module(inputs["in"])

# Handle multi-output nodes (LSTM/GRU return dicts)
if isinstance(raw_output, dict):
    results[node_id] = raw_output
    first_tensor = next(iter(raw_output.values()))
    meta: dict = {
        "outputShape": list(first_tensor.shape),
        "paramCount": sum(p.numel() for p in module.parameters()),
    }
    # ...
    node_results[node_id] = {
        "outputs": {k: tensor_info(v) for k, v in raw_output.items()},
        "metadata": meta,
    }
else:
    results[node_id] = {"out": raw_output}
    # ...
```

**This is the key dispatch.** The `isinstance(raw_output, dict)` check at line 378 is what makes multi-output work. When it is a dict:

1. **`results[node_id]`** is set to the raw dict: `{"out": tensor, "hidden": tensor, "cell": tensor}`. This is what `gather_inputs` reads from.
2. **`node_results[node_id]["outputs"]`** gets `tensor_info` for each key, sent back to the frontend for display.

When it is a plain tensor (the common case), it gets wrapped as `{"out": raw_output}` so downstream routing is uniform.

For our example, after executing the LSTM node, `results["lstm-node-id"]` contains:
```python
{
    "out":    Tensor([32, 10, 128]),   # full output sequence
    "hidden": Tensor([1, 32, 128]),    # final hidden state
    "cell":   Tensor([1, 32, 128]),    # final cell state
}
```

---

## 8. Backend: gather_inputs routes the correct port

### `backend/graph_builder.py`, lines 179-191

```python
def gather_inputs(
    node_id: str, edges: list, results: dict[str, dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """For a given node, collect input tensors from upstream nodes via edges."""
    inputs: dict[str, torch.Tensor] = {}
    for edge in edges:
        if edge["target"]["nodeId"] == node_id:
            src_id = edge["source"]["nodeId"]
            src_port = edge["source"]["portId"]
            tgt_port = edge["target"]["portId"]
            if src_id in results and src_port in results[src_id]:
                inputs[tgt_port] = results[src_id][src_port]
    return inputs
```

This function is the backend mirror of the engine's input gathering (seen in section 4). The logic:

1. Iterate all edges looking for ones targeting the current node.
2. For each edge, read `results[src_id][src_port]` -- the source node's result dict, indexed by the **source port id**.
3. Store it under `tgt_port` -- the **target port id**.

**Concrete trace for our example:**

Suppose we have an edge:
```python
{
    "source": {"nodeId": "lstm-1", "portId": "hidden"},
    "target": {"nodeId": "linear-1", "portId": "in"}
}
```

Then `gather_inputs("linear-1", edges, results)` does:
- `src_id = "lstm-1"`, `src_port = "hidden"`, `tgt_port = "in"`
- `results["lstm-1"]["hidden"]` = the hidden state tensor `[1, 32, 128]`
- Returns `{"in": Tensor([1, 32, 128])}`

The Linear node then runs `module(inputs["in"])` and gets the hidden state as its input. If a different edge connected `portId: "out"` to another Linear, that Linear would instead receive the full output sequence `[32, 10, 128]`.

This is the same `gather_inputs` function used across every execution path in the backend -- forward, training, and inference all call it.

---

## 9. Training loop: how dict outputs are handled

### `backend/graph_builder.py`, `train_graph()`, lines 578-585

Inside the per-batch training loop, layer nodes are processed identically to the forward pass:

```python
# Layer node
inputs = gather_inputs(node_id, edges, batch_results)
if "in" in inputs:
    raw = modules[node_id](inputs["in"])
    if isinstance(raw, dict):
        batch_results[node_id] = raw
    else:
        batch_results[node_id] = {"out": raw}
```

The same `isinstance(raw, dict)` pattern appears here (line 582). When the LSTM module returns a dict, `batch_results[node_id]` gets the full dict. Subsequent calls to `gather_inputs` for downstream nodes will then correctly route each port's tensor.

### Final forward pass after training (lines 690-710)

After the training loop completes, `train_graph` runs one final forward pass through all nodes to get the final-state results to send back to the frontend. The multi-output handling is identical:

```python
elif "in" in inputs:
    raw = modules[node_id](inputs["in"])
    if isinstance(raw, dict):
        final_results[node_id] = raw
        first_t = next(iter(raw.values()))
        node_results[node_id] = {
            "outputs": {k: tensor_info(v) for k, v in raw.items()},
            "metadata": {
                "outputShape": list(first_t.shape),
                "paramCount": sum(p.numel() for p in modules[node_id].parameters()),
            },
        }
    else:
        final_results[node_id] = {"out": raw}
        # ...
```

Note that `node_results` (which gets serialized back to the frontend) includes `tensor_info` for **every** key in the dict, not just `"out"`. The frontend receives shape/mean/std/min/max for the `out`, `hidden`, and `cell` tensors separately.

---

## 10. Inference path: same pattern

### `backend/graph_builder.py`, `infer_graph()`, lines 882-918

Inference uses the trained modules stored in `_model_store["current"]` but follows the exact same dict-detection pattern:

```python
raw = module(inputs["in"])

# Handle multi-output (LSTM/GRU)
if isinstance(raw, dict):
    results[node_id] = raw
    output = next(iter(raw.values()))
else:
    results[node_id] = {"out": raw}
    output = raw
```

Here `results[node_id]` gets the full dict, so downstream `gather_inputs` calls route correctly. The `output` variable (used for metadata like shape and the is-final-layer prediction check) is set to the **first value** in the dict, which is `"out"` -- the full output sequence.

---

## 11. Summary: the naming contract

The entire multi-output system depends on a naming contract. The port id strings must match across four locations:

| Location | File | What it defines |
|----------|------|----------------|
| Port definitions | `src/domain/nodes/layers/lstm.ts:24-29` | Port ids: `"out"`, `"hidden"`, `"cell"` |
| Shape executor outputs | `src/domain/nodes/layers/lstm.ts:55` | `outputs: { out: ..., hidden: ..., cell: ... }` |
| Backend wrapper | `backend/node_builders.py:92` | `return {"out": output, "hidden": hidden, "cell": cell}` |
| Edge source portId | `src/core/graph.ts:60` | `source: { nodeId, portId }` where portId is one of the above |

If any of these four disagree on the string keys, data will silently fail to route. There is no compile-time enforcement of this contract -- it is a runtime convention.

---

## 12. Adding a new multi-output node

To add a new node type that produces multiple outputs (e.g., an Encoder-Decoder that outputs both encoded representation and attention weights):

1. **Frontend definition** -- Add multiple output ports in `getPorts()` with distinct `id` values. Implement the shape executor to return all output shapes in the `outputs` dict, keyed by port id.

2. **Backend wrapper** -- Create a wrapper `nn.Module` whose `forward()` unpacks the underlying module's tuple/named-tuple return value into a `dict[str, torch.Tensor]` keyed by port id.

3. **Backend builder** -- Register the builder in `NODE_BUILDERS` in `backend/node_builders.py`. The builder should return the wrapper module.

No changes are needed to the engine, `gather_inputs`, the training loop, or the inference path. The `isinstance(raw_output, dict)` check in `graph_builder.py` and the edge-based routing in `gather_inputs` handle everything generically.
