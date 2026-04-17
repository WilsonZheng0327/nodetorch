# graph_builder.py — Converts a serialized NodeTorch graph into PyTorch modules and executes them.
#
# Mirrors the TypeScript engine: topological sort → gather inputs → run module → store output.
#
# Main functions:
#   build_and_run_graph() — builds nn.Module for each node, runs a single forward pass.
#                           Returns (modules, tensor_results, display_results, nodes, edges).
#   execute_graph()       — wraps build_and_run_graph with no_grad for the /forward endpoint.
#   train_graph()         — builds modules, then loops: load batch → forward → loss → backward → step.
#                           Streams per-epoch results via on_epoch callback. Supports cancellation.
#   infer_graph()         — uses stored trained modules from _model_store for inference on a single sample.
#
# Node type routing:
#   - DATA_LOADERS: produce tensors directly (MNIST, CIFAR)
#   - OPTIMIZER_NODES: skipped during forward, drive training loop
#   - LOSS_NODES: take named inputs (predictions + labels)
#   - MULTI_INPUT_NODES: take multiple named inputs, called with **kwargs (Add, Concat, MHA, Attention)
#   - SUBGRAPH_TYPE: SubGraphModule wraps inner graph, executes recursively
#   - Everything else: single "in" → module(input) → "out"
#   - Multi-output modules (LSTM/GRU): return dict instead of tensor, stored as-is in results

import math
import torch
import torch.nn as nn

from node_builders import NODE_BUILDERS
from data_loaders import DATA_LOADERS, TRAIN_DATASETS, DENORMALIZERS


# Node types that take multiple named inputs instead of a single "in" port
LOSS_NODES = {"ml.loss.cross_entropy", "ml.loss.mse"}
OPTIMIZER_NODES = {"ml.optimizers.sgd", "ml.optimizers.adam", "ml.optimizers.adamw"}
# Structural nodes with multiple named inputs (passed as **kwargs)
MULTI_INPUT_NODES = {"ml.structural.add", "ml.structural.concat", "ml.layers.multihead_attention", "ml.layers.attention"}
SUBGRAPH_TYPE = "subgraph.block"
SENTINEL_INPUT = "subgraph.input"
SENTINEL_OUTPUT = "subgraph.output"

# --- Device management ---
_device: str = "cpu"

def get_device() -> torch.device:
    return torch.device(_device)

def set_device(device: str):
    global _device
    _device = device

def get_device_name() -> str:
    return _device

# Stores trained modules in memory so inference can reuse them.
# Key: "current" (single session for now), Value: dict of node_id → nn.Module
_model_store: dict[str, dict[str, nn.Module]] = {}

# Cache of the last forward/train/infer pass for layer detail queries.
# Avoids re-running the graph when the user opens the detail modal.
_last_run: dict = {}

def has_trained_model() -> bool:
    return "current" in _model_store

def get_trained_modules() -> dict[str, nn.Module]:
    return _model_store.get("current", {})


def topological_sort(nodes: dict, edges: list) -> list[str]:
    """Kahn's algorithm — same logic as the TypeScript version."""
    downstream: dict[str, list[str]] = {nid: [] for nid in nodes}
    in_degree: dict[str, int] = {nid: 0 for nid in nodes}

    for edge in edges:
        src = edge["source"]["nodeId"]
        tgt = edge["target"]["nodeId"]
        downstream[src].append(tgt)
        in_degree[tgt] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_ids = []

    while queue:
        nid = queue.pop(0)
        sorted_ids.append(nid)
        for child in downstream[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(sorted_ids) != len(nodes):
        raise ValueError("Graph contains a cycle")

    return sorted_ids


class SubGraphModule(nn.Module):
    """Wraps an inner graph's modules into a single nn.Module.
    Executes them in topological order, routing data through edges.

    Key design: inner modules are registered via nn.ModuleDict so their
    parameters are visible to the optimizer during training. The forward()
    method takes **kwargs matching the GraphInput sentinel's port names,
    and returns a dict matching GraphOutput sentinel's port names."""

    def __init__(self, inner_modules: dict[str, nn.Module], inner_nodes: dict,
                 inner_edges: list, inner_order: list[str]):
        super().__init__()
        self.inner_nodes = inner_nodes
        self.inner_edges = inner_edges
        self.inner_order = inner_order
        # Register inner modules so their parameters are visible
        self.inner_modules = nn.ModuleDict()
        for nid, mod in inner_modules.items():
            # Sanitize key for ModuleDict
            # nn.ModuleDict doesn't allow dots or hyphens
            safe_key = nid.replace('.', '_').replace('-', '_')
            self.inner_modules[safe_key] = mod
        # Keep a mapping from node_id → safe_key
        self._key_map = {nid: nid.replace('.', '_').replace('-', '_') for nid in inner_modules}

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, dict[str, torch.Tensor]] = {}
        output_node_id = None

        for node_id in self.inner_order:
            node = self.inner_nodes[node_id]
            node_type = node["type"]

            # Sentinel input: inject parent's inputs
            if node_type == SENTINEL_INPUT:
                results[node_id] = inputs
                continue

            # Sentinel output: save its results, remember its id
            if node_type == SENTINEL_OUTPUT:
                output_node_id = node_id
                node_inputs = gather_inputs(node_id, self.inner_edges, results)
                results[node_id] = node_inputs
                continue

            # Skip optimizers/loss inside subgraph
            if node_type in OPTIMIZER_NODES or node_type in LOSS_NODES:
                continue

            node_inputs = gather_inputs(node_id, self.inner_edges, results)

            # get sanitized safe key for module id
            safe_key = self._key_map.get(node_id)
            if not safe_key or safe_key not in self.inner_modules:
                continue

            module = self.inner_modules[safe_key]

            ### THIS IS THE ACTUAL MODULE EXECUTION ###
            if node_type in MULTI_INPUT_NODES:
                output = module(**{k: v for k, v in node_inputs.items()})
            elif "in" in node_inputs:
                output = module(node_inputs["in"])
            else:
                continue

            results[node_id] = {"out": output}

        # Store inner results for visualization snapshots
        self._last_results = results

        return results.get(output_node_id, {}) if output_node_id else {}


def build_subgraph_module(subgraph_data: dict, parent_input_shapes: dict) -> SubGraphModule:
    """Build a SubGraphModule from a serialized subgraph."""
    nodes = {n["id"]: n for n in subgraph_data["nodes"]}
    edges = subgraph_data["edges"]
    order = topological_sort(nodes, edges)

    # First pass: determine shapes by doing a dry run with shape info
    inner_modules: dict[str, nn.Module] = {}
    shape_results: dict[str, dict[str, list]] = {}

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]
        props = node.get("properties", {})

        if node_type == SENTINEL_INPUT:
            shape_results[node_id] = parent_input_shapes
            continue

        if node_type == SENTINEL_OUTPUT:
            continue

        if node_type in OPTIMIZER_NODES or node_type in LOSS_NODES:
            continue

        # Gather input shapes from upstream
        input_shapes: dict[str, list] = {}
        for edge in edges:
            if edge["target"]["nodeId"] == node_id:
                src_id = edge["source"]["nodeId"]
                src_port = edge["source"]["portId"]
                tgt_port = edge["target"]["portId"]
                if src_id in shape_results and src_port in shape_results[src_id]:
                    input_shapes[tgt_port] = shape_results[src_id][src_port]

        # Nested subgraph: recurse
        if node_type == SUBGRAPH_TYPE:
            sub_data = node.get("subgraph")
            if sub_data and "in" in input_shapes:
                sg_mod = build_subgraph_module(sub_data, {"in": input_shapes["in"]})
                inner_modules[node_id] = sg_mod
                dummy = torch.zeros(input_shapes["in"])
                with torch.no_grad():
                    sg_out = sg_mod(**{"in": dummy})
                first_key = next(iter(sg_out), None)
                if first_key:
                    shape_results[node_id] = {"out": list(sg_out[first_key].shape)}
            continue

        builder = NODE_BUILDERS.get(node_type)
        if not builder:
            continue

        module = builder(props, input_shapes)
        inner_modules[node_id] = module

        # Compute output shape for downstream
        if "in" in input_shapes:
            in_shape = input_shapes["in"]
            dummy = torch.zeros(in_shape)
            with torch.no_grad():
                out = module(dummy)
            if isinstance(out, dict):
                shape_results[node_id] = {k: list(v.shape) for k, v in out.items()}
            else:
                shape_results[node_id] = {"out": list(out.shape)}

    return SubGraphModule(inner_modules, nodes, edges, order)


def gather_inputs(
    node_id: str, edges: list, results: dict[str, dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """For a given node, collect input tensors from upstream nodes via edges.

    Walks all edges, finds those targeting this node, reads the source node's
    output at the specified port. result keys are the TARGET port ids.

    Example: edge {source: {nodeId: "conv", portId: "out"}, target: {nodeId: "relu", portId: "in"}}
    → inputs["in"] = results["conv"]["out"]

    For multi-output nodes (LSTM), results["lstm"] = {"out": tensor, "hidden": tensor, "cell": tensor}
    so the edge's source.portId selects which output to route."""
    inputs: dict[str, torch.Tensor] = {}
    dev = get_device()
    for edge in edges:
        if edge["target"]["nodeId"] == node_id:
            src_id = edge["source"]["nodeId"]
            src_port = edge["source"]["portId"]
            tgt_port = edge["target"]["portId"]
            if src_id in results and src_port in results[src_id]:
                v = results[src_id][src_port]
                # Ensure tensor is on the current device (MPS/CUDA/CPU)
                if isinstance(v, torch.Tensor) and v.device != dev:
                    v = v.to(dev)
                inputs[tgt_port] = v
    return inputs


def build_and_run_graph(graph_data: dict) -> tuple[
    dict[str, nn.Module],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict],
    dict,
    list,
]:
    """
    Build modules and run a forward pass. Returns everything needed
    for both single forward and training.
    """
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    modules: dict[str, nn.Module] = {}
    results: dict[str, dict[str, torch.Tensor]] = {}
    node_results: dict[str, dict] = {}

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]
        props = node["properties"]

        # --- Data nodes ---
        loader = DATA_LOADERS.get(node_type)
        if loader:
            try:
                tensors = loader(props)
                dev = get_device()
                tensors = {k: v.to(dev) for k, v in tensors.items()}
                results[node_id] = tensors
                outputs = {k: tensor_info(v) for k, v in tensors.items()}
                first_tensor = next(iter(tensors.values()))
                node_results[node_id] = {
                    "outputs": outputs,
                    "metadata": {
                        "outputShape": list(first_tensor.shape),
                        "activations": activation_info(first_tensor),
                    },
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Optimizer nodes: skip during forward, handled by training ---
        if node_type in OPTIMIZER_NODES:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {},
            }
            continue

        # --- Loss nodes: take named inputs (predictions + labels) ---
        if node_type in LOSS_NODES:
            inputs = gather_inputs(node_id, edges, results)
            if "predictions" not in inputs or "labels" not in inputs:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "Connect both predictions and labels"},
                }
                continue

            builder = NODE_BUILDERS.get(node_type)
            if not builder:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": f"Unknown node type: {node_type}"},
                }
                continue

            input_shapes = {k: list(v.shape) for k, v in inputs.items()}
            try:
                module = builder(props, input_shapes)
                modules[node_id] = module.to(get_device())
                loss = module(inputs["predictions"], inputs["labels"])
                results[node_id] = {"out": loss}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(loss)},
                    "metadata": {
                        "outputShape": ["scalar"],
                        "lossValue": _safe_float(float(loss.detach())),
                    },
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Structural nodes: multiple named inputs ---
        if node_type in MULTI_INPUT_NODES:
            inputs = gather_inputs(node_id, edges, results)
            builder = NODE_BUILDERS.get(node_type)
            if not builder:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": f"Unknown node type: {node_type}"},
                }
                continue

            input_shapes = {k: list(v.shape) for k, v in inputs.items()}
            try:
                module = builder(props, input_shapes)
                modules[node_id] = module.to(get_device())

                # Pass all inputs as keyword arguments
                output = module(**{k: v for k, v in inputs.items()})
                results[node_id] = {"out": output}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(output)},
                    "metadata": {"outputShape": list(output.shape)},
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Subgraph nodes: build inner modules as a single nn.Module ---
        if node_type == SUBGRAPH_TYPE:
            subgraph_data = node.get("subgraph")
            if not subgraph_data:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "Empty subgraph"},
                }
                continue

            inputs = gather_inputs(node_id, edges, results)
            input_shapes = {k: list(v.shape) for k, v in inputs.items()}
            try:
                sg_module = build_subgraph_module(subgraph_data, input_shapes)
                modules[node_id] = sg_module.to(get_device())

                with torch.no_grad():
                    sg_outputs = sg_module(**inputs)

                # The subgraph returns a dict of port_id → tensor
                # Store the first output for downstream
                first_key = next(iter(sg_outputs), None)
                if first_key:
                    results[node_id] = {"out": sg_outputs[first_key]}
                    meta: dict = {
                        "outputShape": list(sg_outputs[first_key].shape),
                        "paramCount": sum(p.numel() for p in sg_module.parameters()),
                    }
                    # Activation info on the block node itself
                    meta["activations"] = activation_info(sg_outputs[first_key])
                    # Per-inner-node snapshots for visualization
                    inner_snaps = {}
                    inner_results = getattr(sg_module, '_last_results', {})
                    # Sentinel and layer nodes
                    for inner_nid, inner_node in sg_module.inner_nodes.items():
                        inner_type = inner_node["type"]
                        if inner_type == SENTINEL_INPUT:
                            t = inner_results.get(inner_nid, {}).get("in")
                            if t is not None and isinstance(t, torch.Tensor):
                                inner_snaps[inner_nid] = {"activations": activation_info(t)}
                        elif inner_type == SENTINEL_OUTPUT:
                            t = inner_results.get(inner_nid, {}).get("out")
                            if t is not None and isinstance(t, torch.Tensor):
                                inner_snaps[inner_nid] = {"activations": activation_info(t)}
                    for inner_nid, safe_key in sg_module._key_map.items():
                        if safe_key not in sg_module.inner_modules:
                            continue
                        inner_mod = sg_module.inner_modules[safe_key]
                        s: dict = {}
                        wi = module_weight_info(inner_mod)
                        if wi:
                            s["weights"] = wi
                        inner_out = inner_results.get(inner_nid, {}).get("out")
                        if inner_out is not None and isinstance(inner_out, torch.Tensor):
                            s["activations"] = activation_info(inner_out)
                        if s:
                            inner_snaps[inner_nid] = s
                    if inner_snaps:
                        meta["innerSnapshots"] = inner_snaps
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(sg_outputs[first_key])},
                        "metadata": meta,
                    }
                else:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": "Subgraph produced no output"},
                    }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }
            continue

        # --- Layer nodes: single "in" input ---
        builder = NODE_BUILDERS.get(node_type)
        if not builder:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {"error": f"Unknown node type: {node_type}"},
            }
            continue

        inputs = gather_inputs(node_id, edges, results)
        if "in" not in inputs:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {"error": "No input connected"},
            }
            continue

        input_shapes = {k: list(v.shape) for k, v in inputs.items()}
        try:
            module = builder(props, input_shapes)
            modules[node_id] = module.to(get_device())
            raw_output = module(inputs["in"])

            # Handle multi-output nodes (LSTM/GRU return dicts)
            if isinstance(raw_output, dict):
                results[node_id] = raw_output
                first_tensor = next(iter(raw_output.values()))
                meta: dict = {
                    "outputShape": list(first_tensor.shape),
                    "paramCount": sum(p.numel() for p in module.parameters()),
                }
                wi = module_weight_info(module)
                if wi:
                    meta["weights"] = wi
                bi = batchnorm_info(module)
                if bi:
                    meta["batchnorm"] = bi
                meta["activations"] = activation_info(first_tensor)
                node_results[node_id] = {
                    "outputs": {k: tensor_info(v) for k, v in raw_output.items()},
                    "metadata": meta,
                }
            else:
                results[node_id] = {"out": raw_output}
                meta = {
                    "outputShape": list(raw_output.shape),
                    "paramCount": sum(p.numel() for p in module.parameters()),
                }
                wi = module_weight_info(module)
                if wi:
                    meta["weights"] = wi
                bi = batchnorm_info(module)
                if bi:
                    meta["batchnorm"] = bi
                meta["activations"] = activation_info(raw_output)
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(raw_output)},
                    "metadata": meta,
                }
        except Exception as e:
            node_results[node_id] = {
                "outputs": {},
                "metadata": {"error": str(e)},
            }

    # Cache for layer detail queries
    _last_run.clear()
    _last_run["modules"] = modules
    _last_run["results"] = results
    _last_run["nodes"] = nodes
    _last_run["edges"] = edges

    return modules, results, node_results, nodes, edges


def execute_graph(graph_data: dict) -> dict:
    """Run a single forward pass. Returns per-node results."""
    with torch.no_grad():
        _, _, node_results, _, _ = build_and_run_graph(graph_data)
    return node_results


def get_layer_detail(graph_data: dict, node_id: str) -> dict:
    """Return detailed visualization data for a specific node.

    Uses cached results from the last forward/train/infer pass.
    Falls back to running a forward pass if no cache exists.

    Returns:
    - weightMatrix: 2D weight data for heatmap (Linear, Conv2d)
    - featureMaps: per-channel activation grids (Conv2d)
    - attentionMap: attention weight matrix (MHA, Attention)
    - hiddenState: hidden/cell state matrix (LSTM, GRU)
    - confusionData: predictions vs labels (loss nodes)
    """
    # Use cached results if available, otherwise run a forward pass
    if _last_run:
        modules = _last_run["modules"]
        results = _last_run["results"]
        nodes = _last_run["nodes"]
        edges = _last_run["edges"]
    else:
        with torch.no_grad():
            modules, results, _, nodes, edges = build_and_run_graph(graph_data)

    # Resolve module — might be inside a subgraph
    module = modules.get(node_id)
    output = results.get(node_id, {}).get("out")
    node = nodes.get(node_id)

    # Check if node is inside a subgraph
    if not node:
        for nid, mod in modules.items():
            if isinstance(mod, SubGraphModule) and node_id in mod._key_map:
                safe_key = mod._key_map[node_id]
                if safe_key in mod.inner_modules:
                    module = mod.inner_modules[safe_key]
                inner_results = getattr(mod, '_last_results', {})
                output = inner_results.get(node_id, {}).get("out")
                node = mod.inner_nodes.get(node_id)
                break

    if not node:
        return {"error": f"Node {node_id} not found"}

    node_type = node["type"]
    detail: dict = {"nodeType": node_type}

    # --- Weight visualization ---
    if module is not None:
        weight = None
        for name, param in module.named_parameters():
            if 'weight' in name:
                weight = param.detach().cpu().float()
                break
        if weight is not None:
            if weight.dim() == 4:
                # Conv2d: [out_ch, in_ch, kH, kW] — show as kernel grid
                out_ch, in_ch, kH, kW = weight.shape
                n_filters = min(32, out_ch)
                kernels = []
                for f in range(n_filters):
                    # Average across input channels to get one kH x kW image per filter
                    kernel = weight[f].mean(dim=0)  # [kH, kW]
                    # Normalize to 0-255
                    kmin, kmax = float(kernel.min()), float(kernel.max())
                    rng = kmax - kmin if kmax != kmin else 1.0
                    normalized = ((kernel - kmin) / rng * 255).clamp(0, 255).byte()
                    kernels.append(normalized.tolist())
                detail["convKernels"] = {
                    "kernels": kernels,
                    "count": n_filters,
                    "totalFilters": out_ch,
                    "height": kH,
                    "width": kW,
                    "inChannels": in_ch,
                }
            else:
                # Linear or other: 2D weight matrix heatmap
                if weight.dim() == 1:
                    mat = weight.unsqueeze(0)
                elif weight.dim() == 2:
                    mat = weight
                else:
                    mat = weight.reshape(weight.shape[0], -1)

                actual_rows, actual_cols = mat.shape[0], mat.shape[1]
                vmin = _safe_float(float(mat.min()))
                vmax = _safe_float(float(mat.max()))

                # Downsample by block-averaging if too large (max 128x128 for display)
                MAX_DIM = 128
                if mat.shape[0] > MAX_DIM or mat.shape[1] > MAX_DIM:
                    mat = torch.nn.functional.interpolate(
                        mat.unsqueeze(0).unsqueeze(0),
                        size=(min(MAX_DIM, mat.shape[0]), min(MAX_DIM, mat.shape[1])),
                        mode='area',
                    ).squeeze()

                detail["weightMatrix"] = {
                    "data": mat.tolist(),
                    "rows": mat.shape[0],
                    "cols": mat.shape[1],
                    "actualRows": actual_rows,
                    "actualCols": actual_cols,
                    "min": vmin,
                    "max": vmax,
                }

    # --- Feature maps (Conv output channels) ---
    if output is not None and isinstance(output, torch.Tensor) and output.dim() == 4:
        # [batch, channels, H, W] → take first sample, up to 16 channels
        fmaps = output[0].detach().cpu().float()
        n_maps = min(16, fmaps.shape[0])
        maps_list = []
        for c in range(n_maps):
            fm = fmaps[c]
            # Normalize to 0-255
            fmin, fmax = float(fm.min()), float(fm.max())
            rng = fmax - fmin if fmax != fmin else 1.0
            normalized = ((fm - fmin) / rng * 255).clamp(0, 255).byte()
            # Downsample if larger than 32x32
            if normalized.shape[0] > 32 or normalized.shape[1] > 32:
                normalized = torch.nn.functional.interpolate(
                    normalized.unsqueeze(0).unsqueeze(0).float(),
                    size=(min(32, normalized.shape[0]), min(32, normalized.shape[1])),
                    mode='nearest',
                ).squeeze().byte()
            maps_list.append(normalized.tolist())
        detail["featureMaps"] = {
            "maps": maps_list,
            "channels": n_maps,
            "height": len(maps_list[0]),
            "width": len(maps_list[0][0]) if maps_list[0] else 0,
        }

    # --- Attention map ---
    # Re-run MHA/Attention to capture attention weights
    if node_type in ("ml.layers.multihead_attention", "ml.layers.attention"):
        if module is not None:
            inputs = gather_inputs(node_id, edges, results)
            inp = inputs.get("in")
            if inp is not None:
                try:
                    if node_type == "ml.layers.multihead_attention":
                        _, attn_weights = module.mha(inp, inp, inp)
                    else:
                        # Attention module: q=k=v=input
                        q = module.q_proj(inp)
                        k = module.k_proj(inp)
                        d_k = q.shape[-1]
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
                        attn_weights = torch.softmax(scores, dim=-1)
                    if attn_weights is not None:
                        # Take first sample, first head
                        am = attn_weights[0]
                        if am.dim() == 3:
                            am = am[0]  # first head
                        am = am.detach().cpu().float()
                        if am.shape[0] > 64:
                            am = am[:64, :64]
                        detail["attentionMap"] = {
                            "data": am.tolist(),
                            "rows": am.shape[0],
                            "cols": am.shape[1],
                        }
                except Exception:
                    pass  # Attention capture failed, skip

    # --- Hidden state (LSTM/GRU) ---
    if node_type in ("ml.layers.lstm", "ml.layers.gru"):
        node_output = results.get(node_id, {})
        hidden = node_output.get("hidden")
        if hidden is not None and isinstance(hidden, torch.Tensor):
            # [num_layers, batch, hidden_size] → take first sample
            h = hidden[0, 0].detach().cpu().float() if hidden.dim() == 3 else hidden[0].detach().cpu().float()
            detail["hiddenState"] = {
                "data": h.unsqueeze(0).tolist(),
                "rows": 1,
                "cols": len(h),
                "label": "Hidden State",
            }
        cell = node_output.get("cell")
        if cell is not None and isinstance(cell, torch.Tensor):
            c = cell[0, 0].detach().cpu().float() if cell.dim() == 3 else cell[0].detach().cpu().float()
            detail["cellState"] = {
                "data": c.unsqueeze(0).tolist(),
                "rows": 1,
                "cols": len(c),
                "label": "Cell State",
            }

    # --- Confusion matrix (loss nodes) ---
    if node_type in LOSS_NODES:
        # Use full confusion matrix accumulated during training if available
        cached_cm = _last_run.get("confusionMatrix")
        if cached_cm:
            detail["confusionMatrix"] = cached_cm
        else:
            # Fallback: compute from current batch in results
            pred_tensor = None
            label_tensor = None
            for edge in edges:
                if edge["target"]["nodeId"] == node_id:
                    if edge["target"]["portId"] == "predictions":
                        pred_tensor = results.get(edge["source"]["nodeId"], {}).get("out")
                    elif edge["target"]["portId"] == "labels":
                        label_tensor = results.get(edge["source"]["nodeId"], {}).get("labels")
                        if label_tensor is None:
                            label_tensor = results.get(edge["source"]["nodeId"], {}).get("out")
            if pred_tensor is not None and label_tensor is not None:
                preds = pred_tensor.argmax(dim=1).cpu()
                labels = label_tensor.cpu()
                n_classes = max(int(preds.max().item()) + 1, int(labels.max().item()) + 1)
                n_classes = max(n_classes, 2)
                matrix = [[0] * n_classes for _ in range(n_classes)]
                for p, l in zip(preds.tolist(), labels.tolist()):
                    if 0 <= p < n_classes and 0 <= l < n_classes:
                        matrix[l][p] += 1
                detail["confusionMatrix"] = {
                    "data": matrix,
                    "size": n_classes,
                }

    return detail


def train_graph(graph_data: dict, on_epoch=None, on_batch=None, cancel_event=None) -> dict:
    """
    Run a training loop. Finds the optimizer node, builds modules with
    gradient tracking, and trains for the specified number of epochs.

    cancel_event: optional threading.Event — if set, training stops after the current epoch.

    on_epoch: optional callback(epoch, loss, accuracy) for streaming progress.
    """
    graph = graph_data["graph"]
    nodes_list = graph["nodes"]

    # Find the optimizer node
    optimizer_node = None
    for n in nodes_list:
        if n["type"] in OPTIMIZER_NODES:
            optimizer_node = n
            break

    if not optimizer_node:
        return {"error": "No optimizer node in graph"}

    props = optimizer_node["properties"]
    epochs = props.get("epochs", 5)
    lr = props.get("lr", 0.01)
    momentum = props.get("momentum", 0.9)
    weight_decay = props.get("weightDecay", 0)

    # Reproducibility: seed all random number generators before any model building
    seed = props.get("seed", 42)
    if seed is not None:
        import random as _random
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except ImportError:
            pass

    # Find the data node to get batch size
    data_node = None
    for n in nodes_list:
        if DATA_LOADERS.get(n["type"]):
            data_node = n
            break

    if not data_node:
        return {"error": "No data node in graph"}

    batch_size = data_node["properties"].get("batchSize", 1)
    data_loader_fn = DATA_LOADERS[data_node["type"]]

    # Build modules once (first pass establishes shapes)
    modules, results, node_results, nodes, edges = build_and_run_graph(graph_data)

    # Collect all trainable parameters
    all_params = []
    for module in modules.values():
        all_params.extend(module.parameters())

    if not all_params:
        return {"error": "No trainable parameters in graph"}

    # Create optimizer
    opt_type = optimizer_node["type"]
    if opt_type == "ml.optimizers.adam":
        optimizer = torch.optim.Adam(
            all_params,
            lr=lr,
            betas=(props.get("beta1", 0.9), props.get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
    elif opt_type == "ml.optimizers.adamw":
        optimizer = torch.optim.AdamW(
            all_params,
            lr=lr,
            betas=(props.get("beta1", 0.9), props.get("beta2", 0.999)),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            all_params, lr=lr, momentum=momentum, weight_decay=weight_decay,
        )

    # Find loss node
    loss_node_id = None
    for nid, n in nodes.items():
        if n["type"] in LOSS_NODES:
            loss_node_id = nid
            break

    if not loss_node_id:
        return {"error": "No loss node in graph"}

    # Load full dataset for training
    dataset_type = data_node["type"]
    dataset_builder = TRAIN_DATASETS.get(dataset_type)
    if not dataset_builder:
        return {"error": f"Training not supported for dataset: {dataset_type}"}
    # Pass data node properties for datasets that need them (e.g., text vocab_size, max_len)
    data_props = data_node.get("properties", {})
    import inspect
    if inspect.signature(dataset_builder).parameters:
        train_dataset = dataset_builder(**{
            k: data_props[k] for k in inspect.signature(dataset_builder).parameters
            if k in data_props
        })
    else:
        train_dataset = dataset_builder()

    # Split into train/val
    val_split = props.get("valSplit", 0.1)
    val_loader = None
    if val_split > 0 and len(train_dataset) > 10:
        n_val = max(1, int(len(train_dataset) * val_split))
        n_train = len(train_dataset) - n_val
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),  # deterministic split
        )
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

    # Training loop
    import time
    order = topological_sort(nodes, edges)
    epoch_results = []
    total_batches = len(train_loader)

    # Track weight norms for delta computation
    prev_weight_norms: dict[str, float] = {}
    for nid, mod in modules.items():
        params = list(mod.parameters())
        if params:
            prev_weight_norms[nid] = float(torch.cat([p.detach().flatten() for p in params]).float().norm())

    for epoch in range(epochs):
        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            break

        epoch_start = time.time()
        total_loss = 0.0
        correct = 0
        total = 0
        per_class_correct: dict[int, int] = {}
        per_class_total: dict[int, int] = {}
        # Accumulate confusion matrix on last epoch
        is_last_epoch = (epoch == epochs - 1)
        confusion_preds: list[int] = []
        confusion_labels: list[int] = []

        last_batch_results: dict[str, dict[str, torch.Tensor]] = {}

        from tqdm import tqdm
        batch_report_interval = max(1, total_batches // 20)  # ~20 updates per epoch
        for batch_idx, (images, labels) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=80,
        )):
            if cancel_event and cancel_event.is_set():
                break
            dev = get_device()
            images, labels = images.to(dev), labels.to(dev)

            if on_batch and batch_idx % batch_report_interval == 0:
                on_batch({"epoch": epoch + 1, "totalEpochs": epochs, "batch": batch_idx + 1, "totalBatches": total_batches})

            optimizer.zero_grad()

            # Forward pass through the graph
            batch_results: dict[str, dict[str, torch.Tensor]] = {}

            for node_id in order:
                node = nodes[node_id]
                node_type = node["type"]

                # Data node: use current batch
                if DATA_LOADERS.get(node_type):
                    batch_results[node_id] = {"out": images, "labels": labels}
                    continue

                # Optimizer: skip
                if node_type in OPTIMIZER_NODES:
                    continue

                # Loss node
                if node_type in LOSS_NODES:
                    inputs = gather_inputs(node_id, edges, batch_results)
                    if "predictions" in inputs and "labels" in inputs:
                        loss = modules[node_id](inputs["predictions"], inputs["labels"])
                        batch_results[node_id] = {"out": loss}
                    continue

                # Structural nodes (Add, etc.)
                if node_type in MULTI_INPUT_NODES:
                    inputs = gather_inputs(node_id, edges, batch_results)
                    output = modules[node_id](**{k: v for k, v in inputs.items()})
                    batch_results[node_id] = {"out": output}
                    continue

                # Subgraph nodes
                if node_type == SUBGRAPH_TYPE:
                    inputs = gather_inputs(node_id, edges, batch_results)
                    sg_outputs = modules[node_id](**inputs)
                    first_key = next(iter(sg_outputs), None)
                    if first_key:
                        batch_results[node_id] = {"out": sg_outputs[first_key]}
                    continue

                # Layer node
                inputs = gather_inputs(node_id, edges, batch_results)
                if "in" in inputs:
                    raw = modules[node_id](inputs["in"])
                    if isinstance(raw, dict):
                        batch_results[node_id] = raw
                    else:
                        batch_results[node_id] = {"out": raw}

            # Backward pass
            loss_tensor = batch_results.get(loss_node_id, {}).get("out")
            if loss_tensor is not None:
                loss_tensor.backward()
                optimizer.step()
                total_loss += loss_tensor.item()

                # Compute accuracy from the node before loss (predictions)
                # Find which node feeds predictions to the loss node
                for edge in edges:
                    if (edge["target"]["nodeId"] == loss_node_id
                            and edge["target"]["portId"] == "predictions"):
                        pred_node_id = edge["source"]["nodeId"]
                        preds = batch_results.get(pred_node_id, {}).get("out")
                        if preds is not None:
                            predicted = preds.argmax(dim=1)
                            correct += (predicted == labels).sum().item()
                            total += labels.size(0)
                            # Per-class accuracy tracking
                            for cls in labels.unique().tolist():
                                cls = int(cls)
                                mask = labels == cls
                                per_class_total[cls] = per_class_total.get(cls, 0) + mask.sum().item()
                                per_class_correct[cls] = per_class_correct.get(cls, 0) + (predicted[mask] == cls).sum().item()
                            # Accumulate confusion matrix on last epoch
                            if is_last_epoch:
                                confusion_preds.extend(predicted.cpu().tolist())
                                confusion_labels.extend(labels.cpu().tolist())
                        break

            last_batch_results = batch_results

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0

        # --- Validation pass ---
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_total_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    if cancel_event and cancel_event.is_set():
                        break
                    dev = get_device()
                    images, labels = images.to(dev), labels.to(dev)
                    batch_results_v: dict[str, dict[str, torch.Tensor]] = {}
                    for node_id in order:
                        n = nodes[node_id]
                        ntype = n["type"]
                        if DATA_LOADERS.get(ntype):
                            batch_results_v[node_id] = {"out": images, "labels": labels}
                            continue
                        if ntype in OPTIMIZER_NODES:
                            continue
                        mod = modules.get(node_id)
                        if mod is None:
                            continue
                        v_inputs = gather_inputs(node_id, edges, batch_results_v)
                        if ntype in LOSS_NODES:
                            if "predictions" in v_inputs and "labels" in v_inputs:
                                loss_v = mod(v_inputs["predictions"], v_inputs["labels"])
                                batch_results_v[node_id] = {"out": loss_v}
                            continue
                        if ntype in MULTI_INPUT_NODES:
                            out_v = mod(**v_inputs)
                            batch_results_v[node_id] = {"out": out_v}
                            continue
                        if ntype == SUBGRAPH_TYPE:
                            sg = mod(**v_inputs)
                            fk = next(iter(sg), None)
                            if fk:
                                batch_results_v[node_id] = {"out": sg[fk]}
                            continue
                        if "in" in v_inputs:
                            raw = mod(v_inputs["in"])
                            batch_results_v[node_id] = raw if isinstance(raw, dict) else {"out": raw}

                    v_loss_tensor = batch_results_v.get(loss_node_id, {}).get("out")
                    if v_loss_tensor is not None:
                        val_total_loss += float(v_loss_tensor.item())
                        for edge in edges:
                            if (edge["target"]["nodeId"] == loss_node_id
                                    and edge["target"]["portId"] == "predictions"):
                                pnid = edge["source"]["nodeId"]
                                preds_v = batch_results_v.get(pnid, {}).get("out")
                                if preds_v is not None:
                                    pred_v = preds_v.argmax(dim=1)
                                    val_correct += int((pred_v == labels).sum().item())
                                    val_total += int(labels.size(0))
                                break

            if len(val_loader) > 0:
                val_loss = val_total_loss / len(val_loader)
            if val_total > 0:
                val_accuracy = val_correct / val_total

        # Per-node visualization snapshots (weights, gradients, activations)
        node_snapshots = {}
        for node_id, module in modules.items():
            # Recurse into subgraph modules for per-inner-node snapshots
            if isinstance(module, SubGraphModule):
                inner_snaps = {}
                inner_results = getattr(module, '_last_results', {})
                # Sentinel nodes: input gets received distribution, output gets result
                for inner_nid, inner_node in module.inner_nodes.items():
                    inner_type = inner_node["type"]
                    if inner_type == SENTINEL_INPUT:
                        t = inner_results.get(inner_nid, {}).get("in")
                        if t is not None and isinstance(t, torch.Tensor):
                            inner_snaps[inner_nid] = {"activations": activation_info(t)}
                    elif inner_type == SENTINEL_OUTPUT:
                        t = inner_results.get(inner_nid, {}).get("out")
                        if t is not None and isinstance(t, torch.Tensor):
                            inner_snaps[inner_nid] = {"activations": activation_info(t)}
                # Layer nodes: weights, gradients, activations
                for inner_nid, safe_key in module._key_map.items():
                    if safe_key not in module.inner_modules:
                        continue
                    inner_mod = module.inner_modules[safe_key]
                    s = {}
                    wi = module_weight_info(inner_mod)
                    if wi:
                        s["weights"] = wi
                    gi = gradient_info(inner_mod)
                    if gi:
                        s["gradients"] = gi
                    inner_out = inner_results.get(inner_nid, {}).get("out")
                    if inner_out is not None and isinstance(inner_out, torch.Tensor):
                        s["activations"] = activation_info(inner_out)
                    if s:
                        inner_snaps[inner_nid] = s
                # Block node itself gets output activation + inner snapshots
                block_snap: dict = {"innerSnapshots": inner_snaps} if inner_snaps else {}
                out_tensor = last_batch_results.get(node_id, {}).get("out")
                if out_tensor is not None and isinstance(out_tensor, torch.Tensor):
                    block_snap["activations"] = activation_info(out_tensor)
                if block_snap:
                    node_snapshots[node_id] = block_snap
                continue

            snap = {}
            wi = module_weight_info(module)
            if wi:
                snap["weights"] = wi
            gi = gradient_info(module)
            if gi:
                snap["gradients"] = gi
            bi = batchnorm_info(module)
            if bi:
                snap["batchnorm"] = bi
            # Weight delta: change in weight norm since last epoch
            params = list(module.parameters())
            if params and node_id in prev_weight_norms:
                cur_norm = float(torch.cat([p.detach().flatten() for p in params]).float().norm())
                snap["weightDelta"] = _safe_float(abs(cur_norm - prev_weight_norms[node_id]))
                prev_weight_norms[node_id] = cur_norm
            out_tensor = last_batch_results.get(node_id, {}).get("out")
            if out_tensor is not None and isinstance(out_tensor, torch.Tensor):
                snap["activations"] = activation_info(out_tensor)
            if snap:
                node_snapshots[node_id] = snap

        # Gradient flow summary: gradient norm per trainable layer (in topo order)
        # Includes layers inside subgraph blocks (recurses into innerSnapshots).
        grad_entries = []
        for nid in order:
            snap = node_snapshots.get(nid)
            if not snap:
                continue
            if "gradients" in snap:
                short = nodes[nid]["type"].split(".")[-1]
                grad_entries.append((short, snap["gradients"]["norm"]))
            # Recurse into subgraph inner snapshots
            inner = snap.get("innerSnapshots")
            if inner:
                block_prefix = nodes[nid].get("properties", {}).get("blockName") or nodes[nid]["type"].split(".")[-1]
                for inner_nid, inner_snap in inner.items():
                    if "gradients" in inner_snap:
                        inner_short = inner_nid  # e.g. "conv1", "bn1"
                        grad_entries.append((f"{block_prefix}/{inner_short}", inner_snap["gradients"]["norm"]))
        # Count occurrences to decide whether to number
        name_counts: dict[str, int] = {}
        for short, _ in grad_entries:
            name_counts[short] = name_counts.get(short, 0) + 1
        # Build labels: number duplicates (conv2d_1, conv2d_2, ...), leave unique ones as-is
        name_idx: dict[str, int] = {}
        gradient_flow = []
        for short, norm in grad_entries:
            if name_counts[short] > 1:
                name_idx[short] = name_idx.get(short, 0) + 1
                label = f"{short}_{name_idx[short]}"
            else:
                label = short
            gradient_flow.append({"name": label, "norm": norm})

        epoch_result = {
            "epoch": epoch + 1,
            "totalEpochs": epochs,
            "loss": _safe_float(avg_loss),
            "accuracy": _safe_float(accuracy),
            "valLoss": _safe_float(val_loss) if val_loss is not None else None,
            "valAccuracy": _safe_float(val_accuracy) if val_accuracy is not None else None,
            "time": round(epoch_time, 2),
            "batches": total_batches,
            "samples": total,
            "gradientFlow": gradient_flow,
            "perClassAccuracy": sorted([
                {"cls": cls, "accuracy": _safe_float(per_class_correct.get(cls, 0) / per_class_total[cls]) if per_class_total.get(cls, 0) > 0 else 0.0}
                for cls in per_class_total.keys()
            ], key=lambda x: x["accuracy"]) if per_class_total else [],
            "nodeSnapshots": node_snapshots,
        }
        epoch_results.append(epoch_result)

        if on_epoch:
            on_epoch(epoch_result)

    # Final forward pass to get updated node results
    with torch.no_grad():
        final_results: dict[str, dict[str, torch.Tensor]] = {}
        node_results = {}

        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]

            if DATA_LOADERS.get(node_type):
                tensors = data_loader_fn(node["properties"])
                final_results[node_id] = tensors
                outputs = {k: tensor_info(v) for k, v in tensors.items()}
                first_tensor = next(iter(tensors.values()))
                node_results[node_id] = {
                    "outputs": outputs,
                    "metadata": {"outputShape": list(first_tensor.shape)},
                }
                continue

            if node_type in OPTIMIZER_NODES:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {
                        "finalLoss": epoch_results[-1]["loss"] if epoch_results else None,
                        "finalAccuracy": epoch_results[-1]["accuracy"] if epoch_results else None,
                    },
                }
                continue

            if node_type in LOSS_NODES:
                inputs = gather_inputs(node_id, edges, final_results)
                if "predictions" in inputs and "labels" in inputs:
                    loss = modules[node_id](inputs["predictions"], inputs["labels"])
                    final_results[node_id] = {"out": loss}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(loss)},
                        "metadata": {
                            "outputShape": ["scalar"],
                            "lossValue": _safe_float(float(loss.detach())),
                        },
                    }
                continue

            inputs = gather_inputs(node_id, edges, final_results)

            if node_type == SUBGRAPH_TYPE:
                sg_outputs = modules[node_id](**inputs)
                first_key = next(iter(sg_outputs), None)
                if first_key:
                    final_results[node_id] = {"out": sg_outputs[first_key]}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(sg_outputs[first_key])},
                        "metadata": {
                            "outputShape": list(sg_outputs[first_key].shape),
                            "paramCount": sum(p.numel() for p in modules[node_id].parameters()),
                        },
                    }
            elif node_type in MULTI_INPUT_NODES:
                output = modules[node_id](**{k: v for k, v in inputs.items()})
                final_results[node_id] = {"out": output}
                node_results[node_id] = {
                    "outputs": {"out": tensor_info(output)},
                    "metadata": {"outputShape": list(output.shape)},
                }
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
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(raw)},
                        "metadata": {
                            "outputShape": list(raw.shape),
                            "paramCount": sum(p.numel() for p in modules[node_id].parameters()),
                        },
                }

    # Store trained modules for inference
    _model_store["current"] = modules

    # Build full confusion matrix from last epoch and store in cache
    _last_run.clear()
    _last_run["modules"] = modules
    _last_run["results"] = final_results
    _last_run["nodes"] = nodes
    _last_run["edges"] = edges
    if confusion_preds and confusion_labels:
        n_classes = max(max(confusion_preds), max(confusion_labels)) + 1
        matrix = [[0] * n_classes for _ in range(n_classes)]
        for p, l in zip(confusion_preds, confusion_labels):
            if 0 <= p < n_classes and 0 <= l < n_classes:
                matrix[l][p] += 1
        _last_run["confusionMatrix"] = {
            "data": matrix,
            "size": n_classes,
        }

    return {
        "nodeResults": node_results,
        "epochs": epoch_results,
    }


def module_weight_info(module: nn.Module) -> dict | None:
    """Extract weight statistics and histogram bins for visualization."""
    params = list(module.parameters())
    if not params:
        return None

    all_weights = torch.cat([p.detach().flatten() for p in params]).float()
    hist = torch.histogram(all_weights.cpu(), bins=30)

    return {
        "mean": _safe_float(float(all_weights.mean())),
        "std": _safe_float(float(all_weights.std())),
        "min": _safe_float(float(all_weights.min())),
        "max": _safe_float(float(all_weights.max())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
    }


def batchnorm_info(module: nn.Module) -> dict | None:
    """Extract running mean/var histograms from BatchNorm layers."""
    rm = getattr(module, 'running_mean', None)
    rv = getattr(module, 'running_var', None)
    if rm is None or rv is None:
        return None

    rm_f = rm.detach().float().cpu()
    rv_f = rv.detach().float().cpu()
    rm_hist = torch.histogram(rm_f, bins=30)
    rv_hist = torch.histogram(rv_f, bins=30)

    return {
        "runningMean": {
            "mean": _safe_float(float(rm_f.mean())),
            "std": _safe_float(float(rm_f.std())) if rm_f.numel() > 1 else 0.0,
            "histBins": [_safe_float(float(x)) for x in rm_hist.bin_edges[:-1]],
            "histCounts": [int(x) for x in rm_hist.hist],
        },
        "runningVar": {
            "mean": _safe_float(float(rv_f.mean())),
            "std": _safe_float(float(rv_f.std())) if rv_f.numel() > 1 else 0.0,
            "histBins": [_safe_float(float(x)) for x in rv_hist.bin_edges[:-1]],
            "histCounts": [int(x) for x in rv_hist.hist],
        },
    }


def gradient_info(module: nn.Module) -> dict | None:
    """Extract gradient statistics and histogram from the last backward pass."""
    grads = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
    if not grads:
        return None

    all_grads = torch.cat(grads).float()
    hist = torch.histogram(all_grads.cpu(), bins=30)

    return {
        "mean": _safe_float(float(all_grads.mean())),
        "std": _safe_float(float(all_grads.std())) if all_grads.numel() > 1 else 0.0,
        "norm": _safe_float(float(all_grads.norm())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
    }


def activation_info(tensor: torch.Tensor) -> dict:
    """Extract activation statistics and histogram bins for visualization."""
    flat = tensor.detach().flatten().float()
    hist = torch.histogram(flat.cpu(), bins=30)

    return {
        "mean": _safe_float(float(flat.mean())),
        "std": _safe_float(float(flat.std())) if flat.numel() > 1 else 0.0,
        "min": _safe_float(float(flat.min())),
        "max": _safe_float(float(flat.max())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
        "sparsity": _safe_float(float((flat == 0).sum()) / flat.numel()),
    }


def infer_graph(graph_data: dict) -> dict:
    """
    Run inference using trained weights.
    Loads a single sample, runs through stored trained modules,
    returns per-node results + prediction.
    """
    if not has_trained_model():
        return {"error": "No trained model — train first"}

    trained_modules = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    results: dict[str, dict[str, torch.Tensor]] = {}
    node_results: dict[str, dict] = {}
    prediction = None

    with torch.no_grad():
        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]
            props = node["properties"]

            # Data nodes: load a single sample
            loader = DATA_LOADERS.get(node_type)
            if loader:
                try:
                    # Override batch size to 1 for inference
                    infer_props = {**props, "batchSize": 1}
                    tensors = loader(infer_props)
                    tensors = {k: v.to(get_device()) for k, v in tensors.items()}
                    results[node_id] = tensors
                    outputs = {k: tensor_info(v) for k, v in tensors.items()}
                    first_tensor = next(iter(tensors.values()))

                    # Include the actual label and image pixels for display
                    label = int(tensors["labels"][0]) if "labels" in tensors else None

                    # Send raw pixel data for image preview
                    image_data = None
                    image_channels = None
                    if "out" in tensors and tensors["out"].dim() == 4:
                        img = tensors["out"][0]  # [C, H, W]
                        C = img.shape[0]

                        denorm = DENORMALIZERS.get(node_type)
                        if denorm:
                            img = denorm(img)

                        img = (img.clamp(0, 1) * 255).byte()

                        if C == 1:
                            image_data = img[0].tolist()
                            image_channels = 1
                        else:
                            image_data = img.permute(1, 2, 0).tolist()
                            image_channels = C

                    node_results[node_id] = {
                        "outputs": outputs,
                        "metadata": {
                            "outputShape": list(first_tensor.shape),
                            "actualLabel": label,
                            "imagePixels": image_data,
                            "imageChannels": image_channels,
                        },
                    }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
                    }
                continue

            # Optimizer/loss: skip during inference
            if node_type in OPTIMIZER_NODES or node_type in LOSS_NODES:
                node_results[node_id] = {"outputs": {}, "metadata": {}}
                continue

            # Layer / structural nodes: use trained modules
            module = trained_modules.get(node_id)
            if not module:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "No trained module for this node"},
                }
                continue

            inputs = gather_inputs(node_id, edges, results)

            # Subgraph nodes
            if node_type == SUBGRAPH_TYPE:
                try:
                    sg_outputs = module(**inputs)
                    first_key = next(iter(sg_outputs), None)
                    if first_key:
                        results[node_id] = {"out": sg_outputs[first_key]}
                        node_results[node_id] = {
                            "outputs": {"out": tensor_info(sg_outputs[first_key])},
                            "metadata": {"outputShape": list(sg_outputs[first_key].shape)},
                        }
                except Exception as e:
                    node_results[node_id] = {"outputs": {}, "metadata": {"error": str(e)}}
                continue

            # Structural nodes pass all named inputs
            if node_type in MULTI_INPUT_NODES:
                try:
                    output = module(**{k: v for k, v in inputs.items()})
                    results[node_id] = {"out": output}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(output)},
                        "metadata": {"outputShape": list(output.shape)},
                    }
                except Exception as e:
                    node_results[node_id] = {"outputs": {}, "metadata": {"error": str(e)}}
                continue

            if "in" not in inputs:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": "No input connected"},
                }
                continue

            try:
                raw = module(inputs["in"])

                # Handle multi-output (LSTM/GRU)
                if isinstance(raw, dict):
                    results[node_id] = raw
                    output = next(iter(raw.values()))
                else:
                    results[node_id] = {"out": raw}
                    output = raw

                meta: dict = {
                    "outputShape": list(output.shape),
                    "paramCount": sum(p.numel() for p in module.parameters()),
                }

                is_final = any(
                    e["source"]["nodeId"] == node_id
                    and nodes.get(e["target"]["nodeId"], {}).get("type") in LOSS_NODES
                    for e in edges
                )
                if is_final and output.dim() == 2:
                    probs = torch.softmax(output, dim=1)[0]
                    predicted_class = int(probs.argmax())
                    confidence = float(probs[predicted_class])
                    prediction = {
                        "predictedClass": predicted_class,
                        "confidence": _safe_float(confidence),
                        "probabilities": [_safe_float(float(p)) for p in probs],
                    }
                    meta["prediction"] = prediction

                out_info = {k: tensor_info(v) for k, v in raw.items()} if isinstance(raw, dict) else {"out": tensor_info(output)}
                node_results[node_id] = {
                    "outputs": out_info,
                    "metadata": meta,
                }
            except Exception as e:
                node_results[node_id] = {
                    "outputs": {},
                    "metadata": {"error": str(e)},
                }

    # Cache for layer detail queries
    _last_run.clear()
    _last_run["modules"] = trained_modules
    _last_run["results"] = results
    _last_run["nodes"] = nodes
    _last_run["edges"] = edges

    return {
        "nodeResults": node_results,
        "prediction": prediction,
    }


def _safe_float(v: float) -> float | None:
    """Convert to JSON-safe float. NaN and Inf become None."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def tensor_info(t: torch.Tensor) -> dict:
    """Extract displayable info from a tensor (not the raw data)."""
    ft = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
        "min": _safe_float(float(ft.min())),
        "max": _safe_float(float(ft.max())),
    }
