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
import os
import torch
import torch.nn as nn

from node_builders import NODE_BUILDERS
from data_loaders import DATA_LOADERS, TRAIN_DATASETS, DENORMALIZERS


# Node types that take multiple named inputs instead of a single "in" port
LOSS_NODES = {"ml.loss.cross_entropy", "ml.loss.mse"}
OPTIMIZER_NODES = {"ml.optimizers.sgd", "ml.optimizers.adam", "ml.optimizers.adamw"}
# Structural nodes with multiple named inputs (passed as **kwargs)
MULTI_INPUT_NODES = {"ml.structural.add", "ml.structural.concat", "ml.layers.multihead_attention", "ml.layers.attention", "ml.structural.reparameterize", "ml.loss.vae", "ml.loss.gan"}
# All node types recognized as loss functions (for training loop loss detection)
ALL_LOSS_NODES = LOSS_NODES | {"ml.loss.vae", "ml.loss.gan"}
# GAN-specific node types (noise input generates noise, not dataset)
GAN_NOISE_TYPE = "ml.gan.noise_input"
# Diffusion-specific node types
DIFFUSION_SCHEDULER_TYPE = "ml.diffusion.noise_scheduler"
DIFFUSION_EMBED_TYPE = "ml.diffusion.timestep_embed"
SUBGRAPH_TYPE = "subgraph.block"
SENTINEL_INPUT = "subgraph.input"
SENTINEL_OUTPUT = "subgraph.output"

# --- Device management ---
def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

_device: str = _default_device()

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


def save_model(filepath: str = "storage/weights/current.pt") -> dict:
    """Save trained module state dicts to disk."""
    if "current" not in _model_store:
        return {"error": "No trained model to save"}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {nid: mod.state_dict() for nid, mod in _model_store["current"].items()}
    torch.save(state, filepath)
    return {"status": "ok", "path": filepath}


def save_model_bytes() -> bytes | None:
    """Serialize trained module state dicts to bytes for download."""
    if "current" not in _model_store:
        return None
    import io
    buf = io.BytesIO()
    state = {nid: mod.state_dict() for nid, mod in _model_store["current"].items()}
    torch.save(state, buf)
    return buf.getvalue()


def load_model_bytes(graph_data: dict, data: bytes) -> dict:
    """Load state dicts from bytes into freshly built modules."""
    import io
    saved_states = torch.load(io.BytesIO(data), map_location=get_device(), weights_only=True)
    modules = build_modules(graph_data)
    loaded = 0
    for nid, state_dict in saved_states.items():
        if nid in modules:
            modules[nid].load_state_dict(state_dict)
            loaded += 1
    if loaded == 0:
        return {"error": "No matching layers found — graph structure may have changed"}
    _model_store["current"] = modules
    return {"status": "ok"}


def build_modules(graph_data: dict) -> dict[str, nn.Module]:
    """Build modules from the graph without storing results or metadata.

    Runs a minimal forward pass internally (needed to determine input shapes
    for downstream layers) but discards the outputs. Much lighter than
    build_and_run_graph for cases where you only need the module architecture.
    """
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    modules: dict[str, nn.Module] = {}
    results: dict[str, dict] = {}
    dev = get_device()

    with torch.no_grad():
        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]
            props = node.get("properties", {})

            loader = DATA_LOADERS.get(node_type)
            if loader:
                tensors = loader(props)
                results[node_id] = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
                continue

            if node_type in OPTIMIZER_NODES:
                continue

            # GAN noise input: produce dummy noise for shape inference
            if node_type == GAN_NOISE_TYPE:
                builder = NODE_BUILDERS.get(node_type)
                if builder:
                    module = builder(props, {})
                    modules[node_id] = module.to(dev)
                    batch_size = props.get("batchSize", 64)
                    latent_dim = props.get("latentDim", 100)
                    results[node_id] = {"out": torch.randn(batch_size, latent_dim, device=dev)}
                continue

            # Diffusion noise scheduler: build module and produce dummy output
            if node_type == DIFFUSION_SCHEDULER_TYPE:
                inputs = gather_inputs(node_id, edges, results)
                builder = NODE_BUILDERS.get(node_type)
                if builder:
                    input_shapes = {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    module = builder(props, input_shapes)
                    modules[node_id] = module.to(dev)
                    if "images" in inputs:
                        images = inputs["images"]
                        t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=dev)
                        results[node_id] = {"out": images.clone(), "noise": torch.zeros_like(images), "timestep": t_channel}
                continue

            # Diffusion timestep embedding: no input, fixed output
            if node_type == DIFFUSION_EMBED_TYPE:
                builder = NODE_BUILDERS.get(node_type)
                if builder:
                    module = builder(props, {})
                    modules[node_id] = module.to(dev)
                    embed_dim = props.get("embedDim", 128)
                    results[node_id] = {"out": torch.zeros(1, embed_dim, device=dev)}
                continue

            inputs = gather_inputs(node_id, edges, results)
            input_shapes = {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            if node_type == SUBGRAPH_TYPE:
                subgraph_data = node.get("subgraph")
                if subgraph_data:
                    sg_module = build_subgraph_module(subgraph_data, input_shapes)
                    modules[node_id] = sg_module.to(dev)
                    sg_out = sg_module(**inputs)
                    first_key = next(iter(sg_out), None)
                    if first_key:
                        results[node_id] = {"out": sg_out[first_key]}
                continue

            builder = NODE_BUILDERS.get(node_type)
            if not builder:
                continue

            try:
                module = builder(props, input_shapes)
                modules[node_id] = module.to(dev)

                if node_type in LOSS_NODES:
                    if "predictions" in inputs and "labels" in inputs:
                        results[node_id] = {"out": module(inputs["predictions"], inputs["labels"])}
                elif node_type in MULTI_INPUT_NODES:
                    results[node_id] = {"out": module(**{k: v for k, v in inputs.items()})}
                elif "in" in inputs:
                    raw = module(inputs["in"])
                    results[node_id] = raw if isinstance(raw, dict) else {"out": raw}
            except Exception:
                continue

    return modules


def load_model(graph_data: dict, filepath: str = "storage/weights/current.pt") -> dict:
    """Load saved state dicts into freshly built modules from the graph."""
    if not os.path.exists(filepath):
        return {"error": f"No saved model at {filepath}"}
    saved_states = torch.load(filepath, map_location=get_device(), weights_only=True)
    # Build modules from graph (lightweight — no metadata/results)
    modules = build_modules(graph_data)
    # Load saved state dicts
    loaded = 0
    for nid, state_dict in saved_states.items():
        if nid in modules:
            modules[nid].load_state_dict(state_dict)
            loaded += 1
    if loaded == 0:
        return {"error": "No matching layers found — graph structure may have changed"}
    _model_store["current"] = modules
    return {"status": "ok"}


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
            if node_type in OPTIMIZER_NODES or node_type in ALL_LOSS_NODES:
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

        if node_type in OPTIMIZER_NODES or node_type in ALL_LOSS_NODES:
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
                tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
                results[node_id] = tensors
                outputs = {k: tensor_info(v) for k, v in tensors.items() if isinstance(v, torch.Tensor)}
                first_tensor = next(v for v in tensors.values() if isinstance(v, torch.Tensor))
                # Note: dataset nodes do NOT emit an "activations" histogram.
                # "Activation" means the output of a layer's nonlinearity, but
                # a dataset node is just raw (optionally normalized) input data.
                node_results[node_id] = {
                    "outputs": outputs,
                    "metadata": {
                        "outputShape": list(first_tensor.shape),
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

        # --- GAN noise input: produce dummy noise for shape inference ---
        if node_type == GAN_NOISE_TYPE:
            builder = NODE_BUILDERS.get(node_type)
            if builder:
                try:
                    module = builder(props, {})
                    modules[node_id] = module.to(get_device())
                    batch_size = props.get("batchSize", 64)
                    latent_dim = props.get("latentDim", 100)
                    dummy_noise = torch.randn(batch_size, latent_dim, device=get_device())
                    results[node_id] = {"out": dummy_noise}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(dummy_noise)},
                        "metadata": {"outputShape": [batch_size, latent_dim]},
                    }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
                    }
            continue

        # --- Diffusion noise scheduler: passthrough + extra timestep channel ---
        if node_type == DIFFUSION_SCHEDULER_TYPE:
            inputs = gather_inputs(node_id, edges, results)
            builder = NODE_BUILDERS.get(node_type)
            if builder:
                try:
                    input_shapes = {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    module = builder(props, input_shapes)
                    modules[node_id] = module.to(get_device())
                    if "images" in inputs:
                        images = inputs["images"]
                        # Out: noisy images (same shape as input)
                        noisy_out = images.clone()
                        # Noise target: same shape
                        noise_dummy = torch.zeros_like(images)
                        # Timestep channel: [B, 1, H, W]
                        t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)
                        results[node_id] = {"out": noisy_out, "noise": noise_dummy, "timestep": t_channel}
                        node_results[node_id] = {
                            "outputs": {
                                "out": tensor_info(noisy_out),
                                "noise": tensor_info(noise_dummy),
                                "timestep": tensor_info(t_channel),
                            },
                            "metadata": {
                                "outputShape": list(noisy_out.shape),
                                "shapes": [
                                    {"label": "Noisy", "value": list(noisy_out.shape)},
                                    {"label": "Noise", "value": list(noise_dummy.shape)},
                                    {"label": "Timestep", "value": list(t_channel.shape)},
                                ],
                            },
                        }
                    else:
                        node_results[node_id] = {
                            "outputs": {},
                            "metadata": {"error": "No images input connected"},
                        }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
                    }
            continue

        # --- Diffusion timestep embedding: no input, fixed output shape ---
        if node_type == DIFFUSION_EMBED_TYPE:
            builder = NODE_BUILDERS.get(node_type)
            if builder:
                try:
                    module = builder(props, {})
                    modules[node_id] = module.to(get_device())
                    embed_dim = props.get("embedDim", 128)
                    dummy_embed = torch.zeros(1, embed_dim, device=get_device())
                    results[node_id] = {"out": dummy_embed}
                    node_results[node_id] = {
                        "outputs": {"out": tensor_info(dummy_embed)},
                        "metadata": {"outputShape": [1, embed_dim]},
                    }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
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
    # Only show for layers that actually *produce* new feature channels via learned
    # filters. ReLU, Pool, BatchNorm etc. pass 4D tensors through but don't create
    # new features — labeling their output "feature maps" would mislead students.
    FEATURE_MAP_TYPES = {
        "ml.layers.conv2d",
        "ml.layers.conv_transpose2d",
        "ml.layers.pretrained_resnet18",
    }
    if node_type in FEATURE_MAP_TYPES and output is not None and isinstance(output, torch.Tensor) and output.dim() == 4:
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
            # MHA/Attention use query/key/value ports, not "in"
            query = inputs.get("query")
            if query is not None:
                key = inputs.get("key", query)
                value = inputs.get("value", query)
                try:
                    if node_type == "ml.layers.multihead_attention":
                        kwargs = {"need_weights": True, "average_attn_weights": True}
                        if getattr(module, "is_causal", False):
                            seq_len = query.shape[1]
                            kwargs["attn_mask"] = torch.triu(
                                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                                diagonal=1,
                            )
                        _, attn_weights = module.mha(query, key, value, **kwargs)
                    else:
                        # SDPA: compute attention weights manually
                        import math as _math
                        d_k = query.shape[-1]
                        scores = torch.matmul(query, key.transpose(-2, -1)) / _math.sqrt(d_k)
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

    # --- Confusion matrix + misclassifications (loss nodes) ---
    if node_type in LOSS_NODES:
        # Use full confusion matrix accumulated during training if available
        cached_cm = _last_run.get("confusionMatrix")
        if cached_cm:
            detail["confusionMatrix"] = cached_cm
        misclass = _last_run.get("misclassifications")
        if misclass:
            detail["misclassifications"] = misclass
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
            if pred_tensor is not None and label_tensor is not None and pred_tensor.dim() == 2:
                preds = pred_tensor.argmax(dim=1).cpu()
                labels = label_tensor.cpu()
                n_classes = max(int(preds.max().item()) + 1, int(labels.max().item()) + 1)
                n_classes = max(n_classes, 2)
                matrix = [[0] * n_classes for _ in range(n_classes)]
                for p, l in zip(preds.tolist(), labels.tolist()):
                    if 0 <= p < n_classes and 0 <= l < n_classes:
                        matrix[l][p] += 1
                from data_loaders import CLASS_NAMES
                data_type = None
                for nid, nd in _last_run.get("nodes", {}).items():
                    if nd.get("type", "") in DATA_LOADERS:
                        data_type = nd["type"]
                        break
                detail["confusionMatrix"] = {
                    "data": matrix,
                    "size": n_classes,
                    "classNames": CLASS_NAMES.get(data_type, []) if data_type else [],
                }

    return detail


def train_graph(graph_data: dict, on_epoch=None, on_batch=None, cancel_event=None) -> dict:
    """Run a training loop. Delegates to the training plugin system.

    Auto-detects the training paradigm (standard, GAN, diffusion) from node
    types and runs the appropriate loop. See backend/training/ for details.
    """
    from training import run_training
    return run_training(graph_data, on_epoch, on_batch, cancel_event)



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


def evaluate_test_set(graph_data: dict) -> dict:
    """Evaluate trained model on the held-out test set.

    Returns test loss, test accuracy, per-class accuracy, and sample count.
    Only works for classification models (2D predictions).
    """
    if not has_trained_model():
        return {"error": "No trained model — train first"}

    # Diffusion and GAN models don't support standard test evaluation
    graph = graph_data["graph"]
    for n in graph["nodes"]:
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            return {"error": "Diffusion models don't use test set evaluation — use Step Through > Denoise to generate samples"}
        if n["type"] == GAN_NOISE_TYPE:
            return {"error": "GAN models don't use test set evaluation — check generated samples in the training dashboard"}
    from data_loaders import LM_DATASET_TYPES
    for n in graph["nodes"]:
        if n["type"] in LM_DATASET_TYPES:
            return {"error": "Language models don't use test set evaluation — use Step Through > Generate to produce text samples"}

    trained_modules = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Find data, loss, and prediction nodes
    data_node = None
    loss_node_id = None
    pred_node_id = None
    for nid in order:
        n = nodes[nid]
        if n["type"] in DATA_LOADERS:
            data_node = n
        if n["type"] in ALL_LOSS_NODES:
            loss_node_id = nid
            for edge in edges:
                if edge["target"]["nodeId"] == nid and edge["target"]["portId"] == "predictions":
                    pred_node_id = edge["source"]["nodeId"]

    if not data_node:
        return {"error": "No data node in graph"}

    # Load test dataset
    from data_loaders import TEST_DATASETS, CLASS_NAMES
    dataset_type = data_node["type"]
    test_builder = TEST_DATASETS.get(dataset_type)
    if not test_builder:
        return {"error": f"No test set available for {dataset_type}"}

    data_props = data_node.get("properties", {})
    import inspect
    if inspect.signature(test_builder).parameters:
        test_dataset = test_builder(**{
            k: data_props[k] for k in inspect.signature(test_builder).parameters
            if k in data_props
        })
    else:
        test_dataset = test_builder()

    batch_size = data_props.get("batchSize", 32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dev = get_device()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class_correct: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    all_preds: list[int] = []
    all_labels: list[int] = []

    for mod in trained_modules.values():
        if isinstance(mod, nn.Module):
            mod.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(dev), labels.to(dev)

            batch_results: dict[str, dict] = {}
            data_nid = data_node["id"]
            batch_results[data_nid] = {"out": images, "labels": labels}

            for nid in order:
                if nid == data_nid:
                    continue
                n = nodes[nid]
                ntype = n["type"]
                if ntype in OPTIMIZER_NODES:
                    continue
                mod = trained_modules.get(nid)
                if mod is None:
                    continue

                inputs = gather_inputs(nid, edges, batch_results)

                if ntype in LOSS_NODES:
                    if "predictions" in inputs and "labels" in inputs:
                        loss = mod(inputs["predictions"], inputs["labels"])
                        batch_results[nid] = {"out": loss}
                        total_loss += loss.item()
                    continue

                if ntype == SUBGRAPH_TYPE:
                    sg_out = mod(**inputs)
                    first_key = next(iter(sg_out), None)
                    if first_key:
                        batch_results[nid] = {"out": sg_out[first_key]}
                    continue

                if ntype in MULTI_INPUT_NODES:
                    batch_results[nid] = {"out": mod(**{k: v for k, v in inputs.items()})}
                    continue

                if "in" in inputs:
                    raw = mod(inputs["in"])
                    if isinstance(raw, dict):
                        batch_results[nid] = raw
                    else:
                        batch_results[nid] = {"out": raw}

            # Accuracy + confusion data (classification only)
            if pred_node_id and pred_node_id in batch_results:
                preds = batch_results[pred_node_id].get("out")
                if preds is not None and preds.dim() == 2:
                    predicted = preds.argmax(dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    all_preds.extend(predicted.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
                    for cls in labels.unique().tolist():
                        cls = int(cls)
                        mask = labels == cls
                        per_class_total[cls] = per_class_total.get(cls, 0) + mask.sum().item()
                        per_class_correct[cls] = per_class_correct.get(cls, 0) + (predicted[mask] == cls).sum().item()

    # Set modules back to train mode
    for mod in trained_modules.values():
        if isinstance(mod, nn.Module):
            mod.train()

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    accuracy = correct / total if total > 0 else 0

    class_names = CLASS_NAMES.get(dataset_type, [])
    per_class = []
    for cls in sorted(per_class_total.keys()):
        acc = per_class_correct.get(cls, 0) / per_class_total[cls] if per_class_total.get(cls, 0) > 0 else 0
        name = class_names[cls] if cls < len(class_names) else str(cls)
        per_class.append({"cls": cls, "name": name, "accuracy": _safe_float(acc), "count": per_class_total[cls]})

    # Build confusion matrix
    confusion = None
    if all_preds and all_labels:
        n_classes = max(max(all_preds), max(all_labels)) + 1
        matrix = [[0] * n_classes for _ in range(n_classes)]
        for p, l in zip(all_preds, all_labels):
            if 0 <= p < n_classes and 0 <= l < n_classes:
                matrix[l][p] += 1
        confusion = {
            "data": matrix,
            "size": n_classes,
            "classNames": class_names,
        }

    return {
        "testLoss": _safe_float(avg_loss),
        "testAccuracy": _safe_float(accuracy),
        "testSamples": total,
        "perClassAccuracy": per_class,
        "confusionMatrix": confusion,
    }


def infer_graph(graph_data: dict) -> dict:
    """
    Run inference using trained weights.
    Loads a single sample, runs through stored trained modules,
    returns per-node results + prediction.
    """
    if not has_trained_model():
        return {"error": "No trained model — train first"}

    # Diffusion and GAN models don't support standard inference
    graph = graph_data["graph"]
    for n in graph["nodes"]:
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            return {"error": "Diffusion models generate images via denoising — use Step Through > Denoise tab"}
        if n["type"] == GAN_NOISE_TYPE:
            return {"error": "GAN inference generates images from noise — use Step Through > Denoise or check training dashboard for generated samples"}
    from data_loaders import LM_DATASET_TYPES
    for n in graph["nodes"]:
        if n["type"] in LM_DATASET_TYPES:
            return {"error": "Language models generate text autoregressively — use Step Through > Generate tab"}

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
                    tensors = {k: (v.to(get_device()) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
                    results[node_id] = tensors
                    outputs = {k: tensor_info(v) for k, v in tensors.items() if isinstance(v, torch.Tensor)}
                    first_tensor = next(v for v in tensors.values() if isinstance(v, torch.Tensor))

                    # Include the actual label for display (scalar labels only, not sequences)
                    lbl_tensor = tensors.get("labels")
                    label = int(lbl_tensor[0]) if lbl_tensor is not None and isinstance(lbl_tensor, torch.Tensor) and lbl_tensor.dim() == 1 else None

                    meta: dict = {
                        "outputShape": list(first_tensor.shape),
                        "actualLabel": label,
                    }

                    # Image datasets: send raw pixel data for preview
                    if "out" in tensors and isinstance(tensors["out"], torch.Tensor) and tensors["out"].dim() == 4:
                        img = tensors["out"][0].detach().cpu()
                        C = img.shape[0]
                        denorm = DENORMALIZERS.get(node_type)
                        if denorm:
                            img = denorm(img)
                        img = (img.clamp(0, 1) * 255).byte()
                        if C == 1:
                            meta["imagePixels"] = img[0].tolist()
                            meta["imageChannels"] = 1
                        else:
                            meta["imagePixels"] = img.permute(1, 2, 0).tolist()
                            meta["imageChannels"] = C

                    # Text datasets: send raw sample text for preview
                    if "_texts" in tensors:
                        meta["sampleText"] = tensors["_texts"][0][:500]

                    node_results[node_id] = {
                        "outputs": outputs,
                        "metadata": meta,
                    }
                except Exception as e:
                    node_results[node_id] = {
                        "outputs": {},
                        "metadata": {"error": str(e)},
                    }
                continue

            # Optimizer/loss: skip during inference
            if node_type in OPTIMIZER_NODES or node_type in ALL_LOSS_NODES:
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
                    # Classification: show predicted class + probabilities
                    probs = torch.softmax(output, dim=1)[0]
                    predicted_class = int(probs.argmax())
                    confidence = float(probs[predicted_class])
                    prediction = {
                        "predictedClass": predicted_class,
                        "confidence": _safe_float(confidence),
                        "probabilities": [_safe_float(float(p)) for p in probs],
                    }
                    meta["prediction"] = prediction
                elif is_final and output.dim() == 4:
                    # Reconstruction (autoencoder): show output as image
                    img = output[0].detach().cpu()  # [C, H, W]
                    img = (img.clamp(0, 1) * 255).byte()
                    C = img.shape[0]
                    if C == 1:
                        meta["imagePixels"] = img[0].tolist()
                        meta["imageChannels"] = 1
                    else:
                        meta["imagePixels"] = img.permute(1, 2, 0).tolist()
                        meta["imageChannels"] = C
                    meta["reconstruction"] = True

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


def _pick_tracked_samples(dataset, dataset_type: str, n: int = 4) -> list[dict]:
    """Pick N fixed samples from the dataset to track across epochs.

    Returns a list of dicts, each with:
      - idx: index in the dataset
      - label: class label (int)
      - image: pixel data for display (if image dataset)
      - text: raw text (if text dataset)
      - input: the raw input tensor (for forwarding through the model)
    """
    import random
    from data_loaders import DENORMALIZERS

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    samples = []

    for idx in indices:
        item = dataset[idx]
        # Most datasets return (input_tensor, label_tensor)
        if isinstance(item, (tuple, list)) and len(item) == 2:
            inp, lbl = item
        else:
            continue

        sample: dict = {
            "idx": idx,
            "label": int(lbl) if isinstance(lbl, int) or (isinstance(lbl, torch.Tensor) and lbl.dim() == 0) else None,
            "input": inp if isinstance(inp, torch.Tensor) else torch.tensor(inp),
        }

        # Image preview (4D-capable datasets)
        if isinstance(inp, torch.Tensor) and inp.dim() == 3:
            img = inp.detach().cpu()
            denorm = DENORMALIZERS.get(dataset_type)
            if denorm:
                img = denorm(img)
            img = (img.clamp(0, 1) * 255).byte()
            C = img.shape[0]
            if C == 1:
                sample["imagePixels"] = img[0].tolist()
                sample["imageChannels"] = 1
            else:
                sample["imagePixels"] = img.permute(1, 2, 0).tolist()
                sample["imageChannels"] = C

        samples.append(sample)

    return samples


def _probe_tracked_samples(
    tracked: list[dict],
    modules: dict,
    nodes: dict,
    edges: list,
    order: list[str],
    dataset_type: str,
) -> list[dict]:
    """Run tracked samples through the current model and record predictions.

    Returns a list matching the tracked samples, each with:
      - idx, label (from tracked)
      - imagePixels, imageChannels (from tracked, only on first epoch for efficiency)
      - probabilities: softmax output (if classification, i.e. 2D output)
      - predictedClass, confidence
      - loss: per-sample loss value (if available)
      - output: summary of the final layer output (for non-classification models)
    """
    if not tracked:
        return []

    dev = get_device()
    data_nid = None
    loss_nid = None
    for nid in order:
        n = nodes[nid]
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid

    if not data_nid:
        return []

    # Find the node feeding predictions to loss (final layer)
    pred_nid = None
    if loss_nid:
        for edge in edges:
            if (edge["target"]["nodeId"] == loss_nid
                    and edge["target"]["portId"] == "predictions"):
                pred_nid = edge["source"]["nodeId"]
                break

    results = []
    for s in tracked:
        inp = s["input"].unsqueeze(0).to(dev)  # add batch dim
        label_tensor = torch.tensor([s["label"]], dtype=torch.long).to(dev) if s["label"] is not None else None

        # Forward pass through the graph
        batch_results: dict[str, dict] = {}
        batch_results[data_nid] = {"out": inp, "labels": label_tensor}

        with torch.no_grad():
            for nid in order:
                if nid == data_nid:
                    continue
                n = nodes[nid]
                ntype = n["type"]
                if ntype in OPTIMIZER_NODES:
                    continue
                mod = modules.get(nid)
                if mod is None:
                    continue
                mod.eval()

                inputs = gather_inputs(nid, edges, batch_results)

                if ntype in LOSS_NODES:
                    if "predictions" in inputs and "labels" in inputs:
                        try:
                            loss_val = mod(inputs["predictions"], inputs["labels"])
                            batch_results[nid] = {"out": loss_val}
                        except Exception:
                            pass
                    continue

                if ntype == SUBGRAPH_TYPE:
                    try:
                        sg_out = mod(**inputs)
                        first_key = next(iter(sg_out), None)
                        if first_key:
                            batch_results[nid] = {"out": sg_out[first_key]}
                    except Exception:
                        pass
                    continue

                if ntype in MULTI_INPUT_NODES:
                    try:
                        batch_results[nid] = {"out": mod(**{k: v for k, v in inputs.items()})}
                    except Exception:
                        pass
                    continue

                if "in" in inputs:
                    try:
                        raw = mod(inputs["in"])
                        if isinstance(raw, dict):
                            batch_results[nid] = raw
                        else:
                            batch_results[nid] = {"out": raw}
                    except Exception:
                        pass

        # Set modules back to train mode
        for mod in modules.values():
            if isinstance(mod, nn.Module):
                mod.train()

        # Build probe result
        probe: dict = {
            "idx": s["idx"],
            "label": s["label"],
        }

        # Only send image/text on first call (frontend caches it)
        if "imagePixels" in s:
            probe["imagePixels"] = s["imagePixels"]
            probe["imageChannels"] = s.get("imageChannels", 1)

        # Extract prediction from the final layer
        if pred_nid and pred_nid in batch_results:
            pred_out = batch_results[pred_nid].get("out")
            if pred_out is not None and isinstance(pred_out, torch.Tensor):
                if pred_out.dim() == 2:
                    # Classification: softmax probabilities
                    probs = torch.softmax(pred_out, dim=1)[0]
                    probe["probabilities"] = [_safe_float(float(p)) for p in probs.tolist()]
                    pred_class = int(probs.argmax())
                    probe["predictedClass"] = pred_class
                    probe["confidence"] = _safe_float(float(probs[pred_class]))
                else:
                    # Non-classification (autoencoder etc): just report output norm
                    probe["outputNorm"] = _safe_float(float(pred_out.detach().float().norm()))

        # Per-sample loss
        if loss_nid and loss_nid in batch_results:
            loss_out = batch_results[loss_nid].get("out")
            if loss_out is not None and isinstance(loss_out, torch.Tensor):
                probe["loss"] = _safe_float(float(loss_out.item()))

        results.append(probe)

    return results


def _collect_misclassifications(
    images: torch.Tensor,
    predicted: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    dataset_type: str,
    samples: list,
    counts: dict,
    max_per_pair: int = 4,
    max_total: int = 50,
) -> None:
    """Collect a cap of misclassified samples from the current batch.

    Each sample includes displayable pixels (denormalized), predicted/actual labels,
    and softmax confidence for the predicted class. Caps storage so response stays small.
    """
    if len(samples) >= max_total or images.dim() != 4:
        return

    wrong = predicted != labels
    if not wrong.any():
        return

    denorm = DENORMALIZERS.get(dataset_type)
    probs_batch = torch.softmax(logits, dim=1)

    for i in range(labels.size(0)):
        if not bool(wrong[i]):
            continue
        actual = int(labels[i])
        pred = int(predicted[i])
        key = (actual, pred)
        if counts.get(key, 0) >= max_per_pair:
            continue
        if len(samples) >= max_total:
            break
        counts[key] = counts.get(key, 0) + 1

        img = images[i].detach().cpu()
        if denorm:
            img = denorm(img)
        img = (img.clamp(0, 1) * 255).byte()
        C = img.shape[0]
        if C == 1:
            pixels = img[0].tolist()
        else:
            pixels = img.permute(1, 2, 0).tolist()

        samples.append({
            "actual": actual,
            "predicted": pred,
            "confidence": _safe_float(float(probs_batch[i, pred])),
            "imagePixels": pixels,
            "imageChannels": int(C),
        })


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
