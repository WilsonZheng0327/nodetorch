"""Simulate one forward+backward step and return per-node gradient magnitudes.

Used to animate the flow of gradients backward through the graph — educational
visualization of backpropagation.

Runs a single mini-batch, does forward, computes loss, backward, and reads each
node's accumulated gradient norm. Top-level only — SubGraphModule returns the
aggregate norm across all inner params.

Also provides run_backward_step_through() which returns rich per-node backward
stages with gradient visualizations, stats, and educational insights — the
backward counterpart of step_through.run_step_through().
"""

from __future__ import annotations
import copy
import torch

from graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    get_device,
    has_trained_model,
    get_trained_modules,
    _safe_float,
    OPTIMIZER_NODES,
    LOSS_NODES,
    ALL_LOSS_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    DIFFUSION_SCHEDULER_TYPE,
    SubGraphModule,
)
from data_loaders import DATA_LOADERS, DENORMALIZERS
from forward_utils import execute_node
from node_viz import get_backward_viz, compact_stats_with_norm, param_grad_stats


def simulate_backprop(graph_data: dict, batch_size: int = 4) -> dict:
    """Run a single forward+backward pass and return per-node gradient magnitudes
    in topological order. Frontend reverses this for the animation."""
    # Build modules (fresh or trained — doesn't matter much for visualization)
    modules, _, _, nodes, edges = build_and_run_graph(graph_data)

    data_nid = None
    loss_nid = None
    for nid, n in nodes.items():
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid
    if not data_nid or not loss_nid:
        return {"error": "Graph must have a data node and a loss node"}

    order = topological_sort(nodes, edges)

    # Load a small batch
    loader = DATA_LOADERS[nodes[data_nid]["type"]]
    try:
        props = {**nodes[data_nid].get("properties", {}), "batchSize": batch_size}
        tensors = loader(props)
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    dev = get_device()
    images = tensors["out"].to(dev)
    labels = tensors.get("labels")
    if isinstance(labels, torch.Tensor):
        labels = labels.to(dev)

    # Forward pass (NOT under no_grad — we need gradients)
    for m in modules.values():
        m.train()
        m.zero_grad(set_to_none=False)

    batch_results: dict[str, dict] = {data_nid: {"out": images, "labels": labels}}
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
        inputs = gather_inputs(nid, edges, batch_results)
        try:
            out = execute_node(ntype, mod, inputs)
        except Exception as e:
            if ntype in ALL_LOSS_NODES:
                return {"error": f"Loss computation failed: {e}"}
            raise
        if out is not None:
            batch_results[nid] = out

    # Backward
    loss_tensor = batch_results.get(loss_nid, {}).get("out")
    if loss_tensor is None:
        return {"error": "Loss did not produce a value — check predictions/labels connections"}

    try:
        loss_tensor.backward()
    except Exception as e:
        return {"error": f"Backward failed: {e}"}

    # Collect per-top-level-node gradient norms in topological order
    flow = []
    for nid in order:
        n = nodes[nid]
        ntype = n["type"]
        if ntype in DATA_LOADERS or ntype in OPTIMIZER_NODES:
            continue
        mod = modules.get(nid)
        if mod is None:
            continue
        grads = [p.grad.detach().flatten() for p in mod.parameters() if p.grad is not None]
        if grads:
            norm = float(torch.cat(grads).float().norm())
            flow.append({"nodeId": nid, "norm": norm})
        else:
            # Still include non-trainable nodes so animation walks through them
            flow.append({"nodeId": nid, "norm": 0.0})

    return {"flow": flow, "loss": float(loss_tensor.detach().item())}


# --- Rich backward step-through ---

def _friendly_name(node_type: str) -> str:
    """Convert 'ml.layers.conv2d' → 'Conv2d'."""
    last = node_type.split(".")[-1]
    if "_" in last:
        parts = last.split("_")
        return "".join(p.capitalize() for p in parts)
    return last[0].upper() + last[1:]


def _tensor_to_preview_image(img: torch.Tensor, dataset_type: str) -> dict:
    """Convert [C, H, W] to displayable pixels."""
    denorm = DENORMALIZERS.get(dataset_type)
    if denorm:
        img = denorm(img.cpu())
    img = (img.clamp(0, 1) * 255).byte()
    C = img.shape[0]
    if C == 1:
        return {"imagePixels": img[0].tolist(), "imageChannels": 1}
    return {"imagePixels": img.permute(1, 2, 0).tolist(), "imageChannels": C}


def _retain_grad_recursive(modules: dict) -> None:
    """Call retain_grad() on all intermediate tensors inside SubGraphModules.

    Without this, PyTorch discards gradients for non-leaf tensors during backward,
    making it impossible to inspect per-layer gradients inside subgraph blocks.
    """
    for mod in modules.values():
        if not isinstance(mod, SubGraphModule):
            continue
        inner_results = getattr(mod, '_last_results', {})
        for out_dict in inner_results.values():
            if not isinstance(out_dict, dict):
                continue
            for v in out_dict.values():
                if isinstance(v, torch.Tensor) and v.requires_grad and not v.is_leaf:
                    v.retain_grad()
        # Recurse into nested subgraphs
        if hasattr(mod, 'inner_modules'):
            _retain_grad_recursive(dict(mod.inner_modules))


def run_backward_step_through(graph_data: dict) -> dict:
    """Run a forward+backward pass and return rich per-node backward stages.

    Returns: { stages: [...], loss: float, sample: {...}, modelState: {...} }
    """
    graph_data = copy.deepcopy(graph_data)

    # Override batch size to 1
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    # Use trained modules if available
    use_trained = has_trained_model()
    trained_note = None

    if use_trained:
        try:
            modules, nodes, edges, results = _build_with_trained(graph_data)
            trained_note = "using trained weights"
        except Exception:
            use_trained = False
            trained_note = "trained model incompatible — using random weights"

    if not use_trained:
        modules, _, _, nodes, edges = build_and_run_graph(graph_data)
        results = _forward_pass(modules, nodes, edges)
        if trained_note is None:
            trained_note = "using random weights (no trained model)"

    # Retain gradients on all subgraph inner tensors before backward
    _retain_grad_recursive(modules)

    order = topological_sort(nodes, edges)

    # Find data and loss nodes
    data_nid = None
    loss_nid = None
    for nid, n in nodes.items():
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid
    if not data_nid or not loss_nid:
        return {"error": "Graph must have a data node and a loss node"}

    # Extract sample info for the header
    sample_info = {}
    data_tensors = results.get(data_nid, {})
    if data_tensors:
        out = data_tensors.get("out")
        labels = data_tensors.get("labels")
        sample_info = {
            "datasetType": nodes[data_nid]["type"],
            "actualLabel": int(labels[0]) if labels is not None else None,
        }
        if out is not None and isinstance(out, torch.Tensor) and out.dim() == 4:
            sample_info.update(_tensor_to_preview_image(out[0].detach(), nodes[data_nid]["type"]))
        elif out is not None and isinstance(out, torch.Tensor) and out.dim() == 2:
            sample_info["tokenIds"] = out[0].tolist()[:64]
            if "_texts" in data_tensors:
                sample_info["sampleText"] = data_tensors["_texts"][0][:500]

    # Backward pass
    loss_tensor = results.get(loss_nid, {}).get("out")
    if loss_tensor is None:
        return {"error": "Loss did not produce a value — check connections"}

    try:
        loss_tensor.backward()
    except Exception as e:
        return {"error": f"Backward failed: {e}"}

    loss_val = _safe_float(float(loss_tensor.detach().item()))

    # Collect per-node gradient captures from hooks
    grad_captures = getattr(run_backward_step_through, '_grad_captures', {})

    # Build backward stages in reverse topological order
    reversed_order = list(reversed(order))
    stages: list[dict] = []

    for node_id in reversed_order:
        node = nodes[node_id]
        node_type = node["type"]

        if node_type in OPTIMIZER_NODES:
            continue

        # Handle subgraph blocks — flatten inner nodes
        if node_type == SUBGRAPH_TYPE:
            sg_module = modules.get(node_id)
            if isinstance(sg_module, SubGraphModule):
                block_name = node.get("properties", {}).get("blockName") or node_id
                inner_stages = _build_backward_subgraph_stages(
                    sg_module=sg_module,
                    block_name=block_name,
                    depth=1,
                    parent_path=[node_id],
                )
                stages.extend(inner_stages)
                continue

        module = modules.get(node_id)

        # Get activation from forward pass
        out_dict = results.get(node_id, {})
        activation = out_dict.get("out") if isinstance(out_dict, dict) else None

        # Get gradient (from hook or from activation's .grad)
        gradient = None
        if activation is not None and isinstance(activation, torch.Tensor) and activation.grad is not None:
            gradient = activation.grad.detach()
        elif node_id in grad_captures:
            gradient = grad_captures[node_id]

        # Skip data nodes in backward (they don't receive meaningful gradients)
        if node_type in DATA_LOADERS:
            continue

        # Get backward viz from registry
        viz_result = get_backward_viz(node_type, module, activation, gradient)

        stage: dict = {
            "stageId": f"bwd/{node_id}",
            "path": [node_id],
            "nodeId": node_id,
            "nodeType": node_type,
            "displayName": _friendly_name(node_type),
            "depth": 0,
            "inputShape": list(activation.shape) if activation is not None and isinstance(activation, torch.Tensor) else None,
            "outputShape": list(gradient.shape) if gradient is not None else None,
            "gradientShape": list(gradient.shape) if gradient is not None else None,
        }

        # Gradient stats
        if gradient is not None:
            stage["stats"] = compact_stats_with_norm(gradient)

        # Param grad stats (gradient-to-weight ratio)
        pgs = param_grad_stats(module)
        if pgs:
            stage["paramGradStats"] = pgs

        # Viz, extras, insight from registry
        if viz_result.get("viz"):
            stage["viz"] = viz_result["viz"]
        if viz_result.get("extras"):
            stage["extras"] = viz_result["extras"]
        if viz_result.get("insight"):
            stage["insight"] = viz_result["insight"]

        stages.append(stage)

    return {
        "stages": stages,
        "loss": loss_val,
        "sample": sample_info,
        "modelState": {
            "usingTrainedWeights": use_trained,
            "note": trained_note,
        },
    }


def _forward_pass(modules: dict, nodes: dict, edges: list) -> dict:
    """Run a forward pass with gradient tracking (no torch.no_grad)."""
    order = topological_sort(nodes, edges)
    dev = get_device()

    # Set modules to train mode and clear gradients
    for m in modules.values():
        if isinstance(m, torch.nn.Module):
            m.train()
            m.zero_grad(set_to_none=False)

    results: dict[str, dict] = {}

    for nid in order:
        n = nodes[nid]
        ntype = n["type"]

        # Data node: load a batch
        loader = DATA_LOADERS.get(ntype)
        if loader:
            props = n.get("properties", {})
            tensors = loader(props)
            tensors = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
            # Enable grad on float outputs for gradient capture (skip integer tensors like token IDs)
            if "out" in tensors and isinstance(tensors["out"], torch.Tensor) and tensors["out"].is_floating_point():
                tensors["out"] = tensors["out"].detach().requires_grad_(True)
            results[nid] = tensors
            continue

        if ntype in OPTIMIZER_NODES:
            continue

        # Diffusion scheduler: produce 3-port output (noisy, noise, timestep)
        if ntype == DIFFUSION_SCHEDULER_TYPE:
            inputs = gather_inputs(nid, edges, results)
            mod = modules.get(nid)
            if mod is not None and "images" in inputs:
                images = inputs["images"]
                noisy = images.clone().requires_grad_(True)
                noise_dummy = torch.zeros_like(images).requires_grad_(True)
                t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=dev).requires_grad_(True)
                results[nid] = {"out": noisy, "noise": noise_dummy, "timestep": t_channel}
                for v in results[nid].values():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        v.retain_grad()
            continue

        mod = modules.get(nid)
        if mod is None:
            continue

        inputs = gather_inputs(nid, edges, results)
        try:
            out = execute_node(ntype, mod, inputs)
        except Exception:
            continue

        if out is not None:
            # Retain grad on intermediate tensors so we can read them after backward
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    v.retain_grad()
            results[nid] = out

    return results


def _build_with_trained(graph_data: dict) -> tuple:
    """Build modules using trained weights and run forward pass."""
    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)
    dev = get_device()

    modules = {}
    results: dict[str, dict] = {}

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]
        props = node.get("properties", {})

        loader = DATA_LOADERS.get(node_type)
        if loader:
            tensors = loader(props)
            tensors = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
            if "out" in tensors and isinstance(tensors["out"], torch.Tensor) and tensors["out"].is_floating_point():
                tensors["out"] = tensors["out"].detach().requires_grad_(True)
            results[node_id] = tensors
            continue

        if node_type in OPTIMIZER_NODES:
            continue

        # Diffusion scheduler: produce 3-port output
        if node_type == DIFFUSION_SCHEDULER_TYPE:
            module = trained.get(node_id)
            if module is not None:
                modules[node_id] = module
                inputs = gather_inputs(node_id, edges, results)
                if "images" in inputs:
                    images = inputs["images"]
                    noisy = images.clone().requires_grad_(True)
                    noise_dummy = torch.zeros_like(images).requires_grad_(True)
                    t_channel = torch.zeros(images.shape[0], 1, images.shape[2], images.shape[3], device=dev).requires_grad_(True)
                    results[node_id] = {"out": noisy, "noise": noise_dummy, "timestep": t_channel}
                    for v in results[node_id].values():
                        if isinstance(v, torch.Tensor) and v.requires_grad:
                            v.retain_grad()
            continue

        module = trained.get(node_id)
        if module is None:
            raise RuntimeError(f"Trained model missing module for node {node_id}")

        module.train()
        module.zero_grad(set_to_none=False)
        modules[node_id] = module

        inputs = gather_inputs(node_id, edges, results)
        out = execute_node(node_type, module, inputs)
        if out is not None:
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    v.retain_grad()
            results[node_id] = out

    return modules, nodes, edges, results


def _build_backward_subgraph_stages(
    *,
    sg_module: SubGraphModule,
    block_name: str,
    depth: int,
    parent_path: list[str],
) -> list[dict]:
    """Walk a SubGraphModule's inner nodes in reverse and emit backward stages."""
    inner_nodes = sg_module.inner_nodes
    inner_edges = sg_module.inner_edges
    inner_order = sg_module.inner_order
    inner_results = getattr(sg_module, '_last_results', {})

    stages: list[dict] = []
    for inner_nid in reversed(inner_order):
        inner_node = inner_nodes[inner_nid]
        inner_type = inner_node["type"]

        if inner_type in (SENTINEL_INPUT, SENTINEL_OUTPUT):
            continue
        if inner_type in OPTIMIZER_NODES:
            continue

        safe_key = sg_module._key_map.get(inner_nid)
        inner_mod = sg_module.inner_modules[safe_key] if safe_key and safe_key in sg_module.inner_modules else None

        # Nested subgraph
        if isinstance(inner_mod, SubGraphModule):
            nested_name = inner_node.get("properties", {}).get("blockName") or inner_nid
            nested_stages = _build_backward_subgraph_stages(
                sg_module=inner_mod,
                block_name=nested_name,
                depth=depth + 1,
                parent_path=parent_path + [inner_nid],
            )
            stages.extend(nested_stages)
            continue

        # Get activation and gradient
        out_dict = inner_results.get(inner_nid, {})
        activation = out_dict.get("out") if isinstance(out_dict, dict) else None
        gradient = None
        if activation is not None and isinstance(activation, torch.Tensor) and activation.grad is not None:
            gradient = activation.grad.detach()

        viz_result = get_backward_viz(inner_type, inner_mod, activation, gradient)

        stage: dict = {
            "stageId": "/".join(parent_path + [inner_nid]),
            "path": parent_path + [inner_nid],
            "nodeId": inner_nid,
            "nodeType": inner_type,
            "displayName": _friendly_name(inner_type),
            "blockName": block_name,
            "depth": depth,
            "inputShape": list(activation.shape) if activation is not None and isinstance(activation, torch.Tensor) else None,
            "outputShape": list(gradient.shape) if gradient is not None else None,
            "gradientShape": list(gradient.shape) if gradient is not None else None,
        }

        if gradient is not None:
            stage["stats"] = compact_stats_with_norm(gradient)

        pgs = param_grad_stats(inner_mod)
        if pgs:
            stage["paramGradStats"] = pgs

        if viz_result.get("viz"):
            stage["viz"] = viz_result["viz"]
        if viz_result.get("extras"):
            stage["extras"] = viz_result["extras"]
        if viz_result.get("insight"):
            stage["insight"] = viz_result["insight"]

        stages.append(stage)

    return stages
