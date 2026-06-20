"""Single forward pass over a graph.

``build_and_run_graph`` builds an ``nn.Module`` per node, runs one forward
pass, collects per-node display metadata, and caches the run in ``_last_run``
for later layer-detail queries. ``execute_graph`` wraps it in ``no_grad`` for
the /forward endpoint.
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
from engine.graph_builder._state import get_device, _last_run
from engine.graph_builder.build import (
    topological_sort, gather_inputs, build_subgraph_module, SubGraphModule,
)
from engine.graph_builder.stats import (
    tensor_info, activation_info, module_weight_info, batchnorm_info, _safe_float,
)


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
