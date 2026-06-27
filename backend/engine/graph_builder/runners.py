"""Per-node-type execution runners + the dispatch registry.

Each *runner* turns ONE node into raw outputs (stored on the context, for
downstream nodes) plus a frontend ``node_result`` (outputs + display metadata).
``run_node`` looks the runner up by node type — the single dispatch the graph
walks (forward, build, infer, …) share instead of each re-implementing the
per-type branching.

The differences *between* walks live in the ``RunContext``, not in copied code:
  - build a fresh module vs. reuse a trained one  → ``RunContext.get_nn_module``
  - (later) inference batch size / preview metadata → context flags

This module is the backend counterpart of the frontend's per-node ``executors``:
node-type behavior behind a uniform ``run(node, ctx)`` interface, looked up by
type. It mirrors, exactly, the handling that used to be inlined in
``forward.build_and_run_graph``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from engine.graph_builder._state import get_device
from engine.graph_builder.constants import (
    LOSS_NODES, OPTIMIZER_NODES, MULTI_INPUT_NODES,
    GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE,
    SUBGRAPH_TYPE, SENTINEL_INPUT, SENTINEL_OUTPUT,
)
from engine.graph_builder.build import gather_inputs, build_subgraph_module
from engine.graph_builder.stats import (
    tensor_info, _safe_float, module_weight_info, batchnorm_info, activation_info,
)
from engine.node_builders import NODE_BUILDERS, TorchModuleBuilder
from dataprep.data_loaders import DATA_LOADERS

NodeResult = dict


class Runner(Protocol):
    """Executes one node of a given type during a graph walk.

    Each node-type class registers a runner in ``_RUNNERS``; ``run_node``
    dispatches to it. A runner stores the node's raw output tensor(s) in
    ``ctx.results`` (so downstream nodes can read them) and returns the
    ``node_result`` — display outputs + metadata — for the frontend.

    Parameters
    ----------
    node:
        The serialized node dict: ``id``, ``type``, ``properties``, and (for a
        ``subgraph.block``) ``subgraph``.
    ctx:
        The shared ``RunContext`` — graph edges, the ``results`` tensor bus,
        the module store, and the per-walk flags (e.g. ``use_trained``).

    Returns
    -------
    NodeResult
        ``{"outputs": {...}, "metadata": {...}}`` for this node.
    """

    def __call__(self, node: dict, ctx: RunContext) -> NodeResult: ...


@dataclass
class RunContext:
    """Everything a runner needs, plus the knobs that differ between walks."""
    edges: list
    results: dict                                  # node_id -> {port: tensor} (raw, for downstream)
    modules: dict = field(default_factory=dict)    # node_id -> nn.Module (fresh cache OR trained store)
    use_trained: bool = False                      # reuse ctx.modules instead of building

    def get_nn_module(self, node: dict, build: Callable[[], torch.nn.Module]) -> torch.nn.Module:
        """Get this node's nn.Module. In a build walk, construct a fresh one via
        `build` and cache it; in an inference walk (use_trained), reuse the stored
        trained module instead (raising if it isn't there — it never builds).

        This single seam is the only place the forward (build-fresh) and inference
        (reuse-trained) walks differ in how a module is obtained.
        """
        nid = node["id"]
        if self.use_trained:
            module = self.modules.get(nid)
            if module is None:
                raise RuntimeError("No trained module for this node")
            return module
        module = build().to(get_device())
        self.modules[nid] = module
        return module


def _error(msg: str) -> NodeResult:
    return {"outputs": {}, "metadata": {"error": msg}}


def _layer_meta(module: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """Display metadata for a built layer (shared by single- and multi-output layers)."""
    meta: dict = {
        "outputShape": list(tensor.shape),
        "paramCount": sum(p.numel() for p in module.parameters()),
    }
    wi = module_weight_info(module)
    if wi:
        meta["weights"] = wi
    bi = batchnorm_info(module)
    if bi:
        meta["batchnorm"] = bi
    meta["activations"] = activation_info(tensor)
    return meta


# --- Runners (one per node-type class) ---

def run_data(node, ctx: RunContext) -> NodeResult:
    loader = DATA_LOADERS[node["type"]]
    tensors = loader(node["properties"])
    dev = get_device()
    tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
    ctx.results[node["id"]] = tensors
    outputs = {k: tensor_info(v) for k, v in tensors.items() if isinstance(v, torch.Tensor)}
    first_tensor = next(v for v in tensors.values() if isinstance(v, torch.Tensor))
    # Dataset nodes emit no "activations" histogram — they're raw input, not a layer output.
    return {"outputs": outputs, "metadata": {"outputShape": list(first_tensor.shape)}}


def run_skip(node, ctx: RunContext) -> NodeResult:
    # Optimizer nodes: no forward output; the training loop drives them.
    return {"outputs": {}, "metadata": {}}


def run_gan_noise(node, ctx: RunContext) -> NodeResult:
    props = node["properties"]
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return {"outputs": {}, "metadata": {}}
    # Build/register the module for its side effect (it lands in ctx.modules);
    # the output is fabricated noise, not module(...), so we don't keep the handle.
    ctx.get_nn_module(node, lambda: builder(props, input_shapes={}))
    batch_size = props.get("batchSize", 64)
    latent_dim = props.get("latentDim", 100)
    noise = torch.randn(batch_size, latent_dim, device=get_device())
    ctx.results[node["id"]] = {"out": noise}
    return {"outputs": {"out": tensor_info(noise)}, "metadata": {"outputShape": [batch_size, latent_dim]}}


def run_diffusion_scheduler(node, ctx: RunContext) -> NodeResult:
    props = node["properties"]
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return {"outputs": {}, "metadata": {}}
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    input_shapes = {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ctx.get_nn_module(node, lambda: builder(props, input_shapes=input_shapes))
    if "images" not in inputs:
        return _error("No images input connected")
    images = inputs["images"]
    noisy_out = images.clone()
    noise_dummy = torch.zeros_like(images)
    t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)
    ctx.results[node["id"]] = {"out": noisy_out, "noise": noise_dummy, "timestep": t_channel}
    return {
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


def run_diffusion_embed(node, ctx: RunContext) -> NodeResult:
    props = node["properties"]
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return {"outputs": {}, "metadata": {}}
    ctx.get_nn_module(node, lambda: builder(props, input_shapes={}))
    embed_dim = props.get("embedDim", 128)
    embed = torch.zeros(1, embed_dim, device=get_device())
    ctx.results[node["id"]] = {"out": embed}
    return {"outputs": {"out": tensor_info(embed)}, "metadata": {"outputShape": [1, embed_dim]}}


def run_loss(node, ctx: RunContext) -> NodeResult:
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    if "predictions" not in inputs or "labels" not in inputs:
        return _error("Connect both predictions and labels")
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return _error(f"Unknown node type: {node['type']}")
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    module = ctx.get_nn_module(node, lambda: builder(node["properties"], input_shapes=input_shapes))
    loss = module(inputs["predictions"], inputs["labels"])
    ctx.results[node["id"]] = {"out": loss}
    return {
        "outputs": {"out": tensor_info(loss)},
        "metadata": {"outputShape": ["scalar"], "lossValue": _safe_float(float(loss.detach()))},
    }


def run_multi_input(node, ctx: RunContext) -> NodeResult:
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return _error(f"Unknown node type: {node['type']}")
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    module = ctx.get_nn_module(node, lambda: builder(node["properties"], input_shapes=input_shapes))
    output = module(**{k: v for k, v in inputs.items()})
    ctx.results[node["id"]] = {"out": output}
    return {"outputs": {"out": tensor_info(output)}, "metadata": {"outputShape": list(output.shape)}}


def run_subgraph(node, ctx: RunContext) -> NodeResult:
    subgraph_data = node.get("subgraph")
    if not subgraph_data:
        return _error("Empty subgraph")
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    sg_module = ctx.get_nn_module(node, lambda: build_subgraph_module(subgraph_data, input_shapes))

    with torch.no_grad():
        sg_outputs = sg_module(**inputs)

    first_key = next(iter(sg_outputs), None)
    if not first_key:
        return _error("Subgraph produced no output")

    ctx.results[node["id"]] = {"out": sg_outputs[first_key]}
    meta: dict = {
        "outputShape": list(sg_outputs[first_key].shape),
        "paramCount": sum(p.numel() for p in sg_module.parameters()),
        "activations": activation_info(sg_outputs[first_key]),
    }
    # Per-inner-node snapshots for visualization.
    inner_snaps: dict = {}
    inner_results = getattr(sg_module, "_last_results", {})
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
    if inner_snaps:
        meta["innerSnapshots"] = inner_snaps
    return {"outputs": {"out": tensor_info(sg_outputs[first_key])}, "metadata": meta}


def run_layer(node, ctx: RunContext) -> NodeResult:
    """Default: a regular single-input layer (conv, linear, activation, …)."""
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return _error(f"Unknown node type: {node['type']}")
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    if "in" not in inputs:
        return _error("No input connected")
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    module = ctx.get_nn_module(node, lambda: builder(node["properties"], input_shapes=input_shapes))
    raw_output = module(inputs["in"])

    if isinstance(raw_output, dict):  # multi-output nodes (LSTM/GRU return dicts)
        ctx.results[node["id"]] = raw_output
        first_tensor = next(iter(raw_output.values()))
        return {
            "outputs": {k: tensor_info(v) for k, v in raw_output.items()},
            "metadata": _layer_meta(module, first_tensor),
        }
    ctx.results[node["id"]] = {"out": raw_output}
    return {"outputs": {"out": tensor_info(raw_output)}, "metadata": _layer_meta(module, raw_output)}


# --- Dispatch registry ---

_RUNNERS: dict[str, Runner] = {}


def _register(types, runner: Runner) -> None:
    for t in types:
        _RUNNERS[t] = runner


_register(DATA_LOADERS.keys(), run_data)
_register(OPTIMIZER_NODES, run_skip)
_register([GAN_NOISE_TYPE], run_gan_noise)
_register([DIFFUSION_SCHEDULER_TYPE], run_diffusion_scheduler)
_register([DIFFUSION_EMBED_TYPE], run_diffusion_embed)
_register(LOSS_NODES, run_loss)
_register(MULTI_INPUT_NODES, run_multi_input)
_register([SUBGRAPH_TYPE], run_subgraph)


def run_node(node: dict, ctx: RunContext) -> NodeResult:
    """Run one node through its type's runner (default: a regular layer).

    Runtime errors become an ``{"error": …}`` result, mirroring the per-branch
    try/except the inlined dispatch used to have. Guard conditions (missing input,
    unknown type) are returned explicitly by the runners themselves.
    """
    runner = _RUNNERS.get(node["type"], run_layer)
    try:
        return runner(node, ctx)
    except Exception as e:  # noqa: BLE001 — a failing node must not abort the whole walk
        return {"outputs": {}, "metadata": {"error": str(e)}}
