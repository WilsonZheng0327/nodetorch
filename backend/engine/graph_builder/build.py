"""Graph construction primitives.

Topological sort, input routing (``gather_inputs``), composite-block wrapping
(``SubGraphModule`` / ``build_subgraph_module``), and ``build_modules`` — which
builds an ``nn.Module`` per node without retaining results or metadata.
"""

import torch
import torch.nn as nn

from engine.node_builders import NODE_BUILDERS
from dataprep.data_loaders import DATA_LOADERS
from engine.graph_builder.constants import (
    LOSS_NODES, OPTIMIZER_NODES, MULTI_INPUT_NODES, ALL_LOSS_NODES,
    GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE,
    SUBGRAPH_TYPE, SENTINEL_INPUT, SENTINEL_OUTPUT,
)
from engine.graph_builder._state import get_device


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
