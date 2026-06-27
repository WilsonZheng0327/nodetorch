"""Per-node-type execution registry, split into two phases.

A graph walk does the same thing for every node — dispatch on its type, gather
inputs, get its module (build fresh or reuse trained), run it, store the raw
output — and then *separately* formats display metadata for the frontend. Those
two jobs have different audiences:

  - **execute** (this module's ``_EXECUTORS`` registry) is identical for ALL
    walks (forward, build, inference, test, final). It produces an ``Execution``
    and stores the raw tensors on ``ctx.results``. This is the single dispatch —
    so node-type handling can't drift between walks.
  - **describe** is per-walk formatting of an ``Execution`` into a
    ``node_result``. The inspection walk wants full metadata (weights,
    activations); inference wants predictions; ``build_modules`` wants nothing.
    Only this half differs, and it never re-runs a tensor — it just reads the
    ``Execution``.

So a walk is ``describe_X(node, execute(node, ctx))``; ``build_modules`` is just
``execute(node, ctx)``. This is the backend counterpart of the frontend's
per-node ``executors``, with the presentation factored out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from engine.graph_builder._state import get_device
from engine.graph_builder.constants import (
    LOSS_NODES, OPTIMIZER_NODES, MULTI_INPUT_NODES, ALL_LOSS_NODES,
    GAN_NOISE_TYPE, DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE,
    SUBGRAPH_TYPE, SENTINEL_INPUT, SENTINEL_OUTPUT,
)
from engine.graph_builder.build import gather_inputs, build_subgraph_module
from engine.graph_builder.stats import (
    tensor_info, _safe_float, module_weight_info, batchnorm_info, activation_info,
)
from engine.node_builders import NODE_BUILDERS, TorchModuleBuilder
from dataprep.data_loaders import DATA_LOADERS, DENORMALIZERS

NodeResult = dict


@dataclass
class Execution:
    """The raw result of running one node — what every walk shares before
    formatting. The output tensors are already on ``ctx.results``; a ``describe_*``
    presenter turns this into a frontend ``node_result`` without re-running anything.

    Parameters
    ----------
    kind:
        Node category, picked by the executor: ``data``, ``layer``, ``loss``,
        ``multi``, ``subgraph``, ``noise``, ``scheduler``, ``embed``, ``skip``,
        or ``error``.
    module:
        The ``nn.Module`` that ran, if any (``None`` for data / skip / noise /
        embed / errors). Used by presenters for param/weight stats.
    outputs:
        All named raw outputs, ``{port_id: tensor}``.
    primary:
        The main output tensor — used for output shape, activations, predictions
        (``None`` for skip / error).
    extra:
        Type-specific scratch the presenter may need (e.g. ``{"msg": ...}`` for an
        error).
    """
    kind: str
    module: torch.nn.Module | None = None
    outputs: dict = field(default_factory=dict)
    primary: object = None
    extra: dict = field(default_factory=dict)


class NodeExecutor(Protocol):
    """Runs one node of a given type, returning an ``Execution``.

    Registered per node-type class in ``_EXECUTORS`` and dispatched by ``execute``.
    An executor gathers inputs, obtains the node's module (``ctx.get_nn_module``),
    runs it, stores the raw output(s) on ``ctx.results``, and returns an
    ``Execution`` describing what ran. It does NOT build display metadata — that's
    the ``describe_*`` half.
    """

    def __call__(self, node: dict, ctx: RunContext) -> Execution: ...


@dataclass
class RunContext:
    """Everything an executor needs, plus the knobs that differ between walks.

    ``use_trained`` / ``batch_override`` / ``skip_losses`` are the only behavioral
    differences between the build-fresh inspection walk and the trained inference
    walks — the per-node dispatch itself is shared.
    """
    edges: list
    results: dict                                  # node_id -> {port: tensor} (raw, for downstream)
    modules: dict = field(default_factory=dict)    # node_id -> nn.Module (fresh cache OR trained store)
    use_trained: bool = False                      # reuse ctx.modules instead of building
    batch_override: int | None = None              # force a batchSize on data loaders (inference: 1)
    skip_losses: bool = False                      # don't run loss nodes (inference / test display)

    def get_nn_module(self, node: dict, build) -> torch.nn.Module:
        """Get this node's nn.Module. In a build walk, construct a fresh one via
        ``build`` and cache it; in an inference walk (``use_trained``), reuse the
        stored trained module instead (raising if it isn't there — it never builds).

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


def _err(msg: str) -> Execution:
    return Execution(kind="error", extra={"msg": msg})


# --- Executors (one per node-type class) — produce an Execution, store raw outputs ---

def execute_data(node, ctx: RunContext) -> Execution:
    props = node["properties"]
    if ctx.batch_override is not None:
        props = {**props, "batchSize": ctx.batch_override}
    loader = DATA_LOADERS[node["type"]]
    tensors = loader(props)
    dev = get_device()
    tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
    ctx.results[node["id"]] = tensors
    primary = next(v for v in tensors.values() if isinstance(v, torch.Tensor))
    return Execution(kind="data", outputs=tensors, primary=primary)


def execute_skip(node, ctx: RunContext) -> Execution:
    # Optimizer nodes: no forward output; the training loop drives them.
    return Execution(kind="skip")


def execute_gan_noise(node, ctx: RunContext) -> Execution:
    props = node["properties"]
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return Execution(kind="skip")
    # Build/register the module for its side effect (it lands in ctx.modules);
    # the output is fabricated noise, not module(...), so we don't keep the handle.
    ctx.get_nn_module(node, lambda: builder(props, input_shapes={}))
    batch_size = props.get("batchSize", 64)
    latent_dim = props.get("latentDim", 100)
    noise = torch.randn(batch_size, latent_dim, device=get_device())
    ctx.results[node["id"]] = {"out": noise}
    return Execution(kind="noise", outputs={"out": noise}, primary=noise)


def execute_diffusion_scheduler(node, ctx: RunContext) -> Execution:
    props = node["properties"]
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return Execution(kind="skip")
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    input_shapes = {k: list(v.shape) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    ctx.get_nn_module(node, lambda: builder(props, input_shapes=input_shapes))
    if "images" not in inputs:
        return _err("No images input connected")
    images = inputs["images"]
    noisy_out = images.clone()
    noise_dummy = torch.zeros_like(images)
    t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=images.device)
    outputs = {"out": noisy_out, "noise": noise_dummy, "timestep": t_channel}
    ctx.results[node["id"]] = outputs
    return Execution(kind="scheduler", outputs=outputs, primary=noisy_out)


def execute_diffusion_embed(node, ctx: RunContext) -> Execution:
    props = node["properties"]
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return Execution(kind="skip")
    ctx.get_nn_module(node, lambda: builder(props, input_shapes={}))
    embed_dim = props.get("embedDim", 128)
    embed = torch.zeros(1, embed_dim, device=get_device())
    ctx.results[node["id"]] = {"out": embed}
    return Execution(kind="embed", outputs={"out": embed}, primary=embed)


def execute_loss(node, ctx: RunContext) -> Execution:
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    if "predictions" not in inputs or "labels" not in inputs:
        return _err("Connect both predictions and labels")
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return _err(f"Unknown node type: {node['type']}")
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    module = ctx.get_nn_module(node, lambda: builder(node["properties"], input_shapes=input_shapes))
    loss = module(inputs["predictions"], inputs["labels"])
    ctx.results[node["id"]] = {"out": loss}
    return Execution(kind="loss", module=module, outputs={"out": loss}, primary=loss)


def execute_multi_input(node, ctx: RunContext) -> Execution:
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return _err(f"Unknown node type: {node['type']}")
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    module = ctx.get_nn_module(node, lambda: builder(node["properties"], input_shapes=input_shapes))
    output = module(**{k: v for k, v in inputs.items()})
    ctx.results[node["id"]] = {"out": output}
    return Execution(kind="multi", module=module, outputs={"out": output}, primary=output)


def execute_subgraph(node, ctx: RunContext) -> Execution:
    subgraph_data = node.get("subgraph")
    if not subgraph_data:
        return _err("Empty subgraph")
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    sg_module = ctx.get_nn_module(node, lambda: build_subgraph_module(subgraph_data, input_shapes))

    with torch.no_grad():
        sg_outputs = sg_module(**inputs)

    first_key = next(iter(sg_outputs), None)
    if not first_key:
        return _err("Subgraph produced no output")

    primary = sg_outputs[first_key]
    ctx.results[node["id"]] = {"out": primary}
    return Execution(kind="subgraph", module=sg_module, outputs={"out": primary}, primary=primary)


def execute_layer(node, ctx: RunContext) -> Execution:
    """Default: a regular single-input layer (conv, linear, activation, …)."""
    builder: TorchModuleBuilder | None = NODE_BUILDERS.get(node["type"])
    if not builder:
        return _err(f"Unknown node type: {node['type']}")
    inputs = gather_inputs(node["id"], ctx.edges, ctx.results)
    if "in" not in inputs:
        return _err("No input connected")
    input_shapes = {k: list(v.shape) for k, v in inputs.items()}
    module = ctx.get_nn_module(node, lambda: builder(node["properties"], input_shapes=input_shapes))
    raw = module(inputs["in"])

    if isinstance(raw, dict):  # multi-output nodes (LSTM/GRU return dicts)
        ctx.results[node["id"]] = raw
        return Execution(kind="layer", module=module, outputs=raw, primary=next(iter(raw.values())))
    ctx.results[node["id"]] = {"out": raw}
    return Execution(kind="layer", module=module, outputs={"out": raw}, primary=raw)


# --- Dispatch registry (the single, shared per-type dispatch) ---

_EXECUTORS: dict[str, NodeExecutor] = {}


def _register(types, executor: NodeExecutor) -> None:
    for t in types:
        _EXECUTORS[t] = executor


_register(DATA_LOADERS.keys(), execute_data)
_register(OPTIMIZER_NODES, execute_skip)
_register([GAN_NOISE_TYPE], execute_gan_noise)
_register([DIFFUSION_SCHEDULER_TYPE], execute_diffusion_scheduler)
_register([DIFFUSION_EMBED_TYPE], execute_diffusion_embed)
_register(LOSS_NODES, execute_loss)
_register(MULTI_INPUT_NODES, execute_multi_input)
_register([SUBGRAPH_TYPE], execute_subgraph)


def execute(node: dict, ctx: RunContext) -> Execution:
    """Run one node through its type's executor (default: a regular layer).

    Stores the node's raw output(s) on ``ctx.results`` and returns an
    ``Execution``. Runtime errors become ``Execution(kind="error")`` — mirroring
    the per-branch try/except the inlined dispatch used to have — so one bad node
    can't abort the whole walk. Guard conditions (missing input, unknown type) are
    returned explicitly by the executors as ``error`` Executions.
    """
    if ctx.skip_losses and node["type"] in ALL_LOSS_NODES:
        return Execution(kind="skip")
    executor = _EXECUTORS.get(node["type"], execute_layer)
    try:
        return executor(node, ctx)
    except Exception as e:  # noqa: BLE001 — a failing node must not abort the walk
        return _err(str(e))


# --- Presenters: format an Execution into a frontend node_result ---

def _tensor_infos(outputs: dict) -> dict:
    return {k: tensor_info(v) for k, v in outputs.items() if isinstance(v, torch.Tensor)}


def _layer_meta(module: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """Full layer metadata: shape, params, weight/batchnorm stats, activations."""
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


def _inner_snapshots(sg_module) -> dict:
    """Per-inner-node activation snapshots for a subgraph block (for the viz)."""
    snaps: dict = {}
    inner_results = getattr(sg_module, "_last_results", {})
    for inner_nid, inner_node in sg_module.inner_nodes.items():
        inner_type = inner_node["type"]
        if inner_type == SENTINEL_INPUT:
            t = inner_results.get(inner_nid, {}).get("in")
        elif inner_type == SENTINEL_OUTPUT:
            t = inner_results.get(inner_nid, {}).get("out")
        else:
            continue
        if t is not None and isinstance(t, torch.Tensor):
            snaps[inner_nid] = {"activations": activation_info(t)}
    return snaps


def describe_inspection(node: dict, exe: Execution) -> NodeResult:
    """Inspection-walk presentation: full display metadata for each node
    (weights, activations, batchnorm, inner snapshots). The standalone
    build-fresh-and-inspect pass; not the training forward."""
    if exe.kind == "error":
        return {"outputs": {}, "metadata": {"error": exe.extra["msg"]}}
    if exe.kind == "skip":
        return {"outputs": {}, "metadata": {}}

    outputs = _tensor_infos(exe.outputs)

    if exe.kind in ("data", "noise", "embed", "multi"):
        return {"outputs": outputs, "metadata": {"outputShape": list(exe.primary.shape)}}

    if exe.kind == "scheduler":
        o = exe.outputs
        return {"outputs": outputs, "metadata": {
            "outputShape": list(exe.primary.shape),
            "shapes": [
                {"label": "Noisy", "value": list(o["out"].shape)},
                {"label": "Noise", "value": list(o["noise"].shape)},
                {"label": "Timestep", "value": list(o["timestep"].shape)},
            ],
        }}

    if exe.kind == "loss":
        return {"outputs": outputs, "metadata": {
            "outputShape": ["scalar"],
            "lossValue": _safe_float(float(exe.primary.detach())),
        }}

    if exe.kind == "subgraph":
        meta: dict = {
            "outputShape": list(exe.primary.shape),
            "paramCount": sum(p.numel() for p in exe.module.parameters()),
            "activations": activation_info(exe.primary),
        }
        snaps = _inner_snapshots(exe.module)
        if snaps:
            meta["innerSnapshots"] = snaps
        return {"outputs": outputs, "metadata": meta}

    # layer (default)
    return {"outputs": outputs, "metadata": _layer_meta(exe.module, exe.primary)}


def inspect_node(node: dict, ctx: RunContext) -> NodeResult:
    """Run a node and format it for the standalone inspection walk
    (execute + describe_inspection). This is one specific composition — the
    universal per-node step is ``execute``, not this."""
    return describe_inspection(node, execute(node, ctx))


# --- Inference presentation: prediction / reconstruction / single-sample preview ---

def _scalar_label(lbl) -> int | None:
    """The sample's class label, for display — only scalar (1D) labels, not sequences."""
    if isinstance(lbl, torch.Tensor) and lbl.dim() == 1:
        return int(lbl[0])
    return None


def _add_image_preview(meta: dict, img: torch.Tensor, node_type: str) -> None:
    """Attach displayable pixels for an image tensor [C, H, W]. Denormalizes when
    the node has a registered denormalizer (data nodes); raw clamp otherwise
    (reconstruction outputs, whose layer type isn't in DENORMALIZERS)."""
    img = img.detach().cpu()
    denorm = DENORMALIZERS.get(node_type)
    if denorm:
        img = denorm(img)
    img = (img.clamp(0, 1) * 255).byte()
    channels = img.shape[0]
    if channels == 1:
        meta["imagePixels"] = img[0].tolist()
        meta["imageChannels"] = 1
    else:
        meta["imagePixels"] = img.permute(1, 2, 0).tolist()
        meta["imageChannels"] = channels


def describe_inference(node: dict, exe: Execution, edges: list, nodes: dict) -> NodeResult:
    """Inference-walk presentation: single-sample run on trained modules.

    Data nodes show the actual label + an image/text preview; the final layer
    (the one feeding the loss) shows a class prediction (2D) or a reconstruction
    image (4D). Losses/optimizers are skipped upstream (``skip_losses``), so they
    arrive as ``kind="skip"`` and render empty.
    """
    if exe.kind == "error":
        return {"outputs": {}, "metadata": {"error": exe.extra["msg"]}}
    if exe.kind == "skip":
        return {"outputs": {}, "metadata": {}}

    outputs = _tensor_infos(exe.outputs)
    node_type = node["type"]

    if exe.kind == "data":
        tensors = exe.outputs
        meta: dict = {
            "outputShape": list(exe.primary.shape),
            "actualLabel": _scalar_label(tensors.get("labels")),
        }
        out = tensors.get("out")
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            _add_image_preview(meta, out[0], node_type)
        if "_texts" in tensors:
            meta["sampleText"] = tensors["_texts"][0][:500]
        return {"outputs": outputs, "metadata": meta}

    if exe.kind in ("subgraph", "multi"):
        return {"outputs": outputs, "metadata": {"outputShape": list(exe.primary.shape)}}

    # layer (default): shape + params, plus prediction/reconstruction if it's the head
    output = exe.primary
    meta = {
        "outputShape": list(output.shape),
        "paramCount": sum(p.numel() for p in exe.module.parameters()),
    }
    is_final = any(
        e["source"]["nodeId"] == node["id"]
        and nodes.get(e["target"]["nodeId"], {}).get("type") in LOSS_NODES
        for e in edges
    )
    if is_final and output.dim() == 2:
        probs = torch.softmax(output, dim=1)[0]
        predicted_class = int(probs.argmax())
        meta["prediction"] = {
            "predictedClass": predicted_class,
            "confidence": _safe_float(float(probs[predicted_class])),
            "probabilities": [_safe_float(float(p)) for p in probs],
        }
    elif is_final and output.dim() == 4:
        _add_image_preview(meta, output[0], node_type)
        meta["reconstruction"] = True
    return {"outputs": outputs, "metadata": meta}
