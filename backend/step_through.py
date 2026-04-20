"""Step-through — run a forward pass and record intermediate activations at each node.

Returns an ordered list of "stages" — one per node executed. Each stage has:
  - identification: stageId, path (hierarchical), nodeId, nodeType, displayName, depth
  - shapes: inputShape, outputShape
  - stats: histogram + mean/std/min/max/sparsity of the output
  - viz: optional visualization data (image/feature_maps/vector/probabilities/scalar)
  - insight: plain-English explanation of what this layer did

The schema is designed to be forward-compatible:
  - `path` is a list so subgraph recursion (Phase 2) just extends it
  - `depth` indicates nesting level for indented display
  - `viz.kind` is a discriminator so new visualization types can be added
  - Optional fields are handled gracefully by the frontend

Kept deliberately independent from graph_builder.py — only uses it to build modules.
"""

from __future__ import annotations
import copy
import torch

from graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    has_trained_model,
    get_trained_modules,
    get_device,
    _safe_float,
    LOSS_NODES,
    ALL_LOSS_NODES,
    OPTIMIZER_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    SubGraphModule,
)
from forward_utils import execute_node
from data_loaders import DATA_LOADERS, DENORMALIZERS
from node_viz import get_forward_viz, compact_stats



# --- Main entry point ---

def run_step_through(graph_data: dict, sample_idx: int | None = None, mask: list | None = None) -> dict:
    """Run a forward pass and return ordered stages.

    Args:
        graph_data: serialized NodeTorch graph (same format as /forward)
        sample_idx: optional index into the dataset (not used in Phase 1; random batch used)
        mask: optional 2D [H, W] binary mask (1 = zero out pixel) for input perturbation

    Returns:
        { "stages": [...], "sample": {...}, "modelState": {...} }
    """
    # Deep-copy so we can override batch size without mutating caller's data
    graph_data = copy.deepcopy(graph_data)

    # Override batch size to 1 for step-through (showing one sample at a time)
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    # Use trained modules if available (much more educational than random weights)
    use_trained = has_trained_model()
    trained_note = None

    if use_trained:
        try:
            modules, results, nodes, edges = _forward_with_trained(graph_data, mask=mask)
            trained_note = "using trained weights"
        except Exception:
            # Fall back to fresh build if trained modules don't match graph
            use_trained = False
            trained_note = "trained model incompatible — using random weights (retrain after graph changes)"

    if not use_trained:
        with torch.no_grad():
            modules, results, _, nodes, edges = build_and_run_graph(graph_data)
        # Apply mask to the data node's output if provided
        if mask is not None:
            _apply_mask_to_data(results, nodes, mask)
            # Re-run downstream nodes with the masked input
            _rerun_from_data(modules, results, nodes, edges)
        if trained_note is None:
            trained_note = "using random weights (no trained model)"

    order = topological_sort(nodes, edges)

    # Build stages
    stages: list[dict] = []
    sample_info = _extract_sample_info(nodes, results)

    for node_id in order:
        node = nodes[node_id]
        node_type = node["type"]

        # Skip optimizer nodes (no tensor transformation to visualize)
        if node_type in OPTIMIZER_NODES:
            continue

        # Subgraph block: recurse into inner modules, emit one stage per inner layer
        if node_type == SUBGRAPH_TYPE:
            sg_module = modules.get(node_id)
            if isinstance(sg_module, SubGraphModule):
                block_name = node.get("properties", {}).get("blockName") or node_id
                inner_stages = _build_subgraph_stages(
                    sg_module=sg_module,
                    block_name=block_name,
                    depth=1,
                    parent_path=[node_id],
                )
                stages.extend(inner_stages)
                continue

        stage = _build_stage(
            node=node,
            node_id=node_id,
            path=[node_id],
            depth=0,
            edges=edges,
            results=results,
            nodes=nodes,
            module=modules.get(node_id),
        )
        if stage:
            stages.append(stage)

    # Compute saliency map (if possible) and attach to the data node stage
    saliency = _compute_saliency(modules, nodes, edges)
    if saliency is not None:
        for st in stages:
            node_type_st = st.get("nodeType")
            if node_type_st in DATA_LOADERS:
                extras = st.get("extras", [])
                extras.append(saliency)
                st["extras"] = extras
                break

    return {
        "stages": stages,
        "sample": sample_info,
        "modelState": {
            "usingTrainedWeights": use_trained,
            "note": trained_note,
        },
    }


# --- Input perturbation (mask) ---

def _apply_mask_to_tensors(tensors: dict, mask: list) -> None:
    """Zero out pixels in the data tensors where mask is 1.
    Mutates tensors in place. Mask is a 2D [H, W] array of 0/1."""
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    for key, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            continue
        if v.dim() == 4 and v.shape[-2] == mask_tensor.shape[0] and v.shape[-1] == mask_tensor.shape[1]:
            # [B, C, H, W] — broadcast mask to [1, 1, H, W]
            m = mask_tensor.to(v.device).unsqueeze(0).unsqueeze(0)
            tensors[key] = v * (1 - m)
        elif v.dim() == 3 and v.shape[-2] == mask_tensor.shape[0] and v.shape[-1] == mask_tensor.shape[1]:
            m = mask_tensor.to(v.device).unsqueeze(0)
            tensors[key] = v * (1 - m)


def _apply_mask_to_data(results: dict, nodes: dict, mask: list) -> None:
    """Find the data node in results and apply mask to its tensors."""
    for nid, node in nodes.items():
        if node["type"] in DATA_LOADERS and nid in results:
            _apply_mask_to_tensors(results[nid], mask)
            return


def _rerun_from_data(modules: dict, results: dict, nodes: dict, edges: list) -> None:
    """After masking the data node, re-run downstream modules to refresh their outputs."""
    order = topological_sort(nodes, edges)
    with torch.no_grad():
        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]

            # Skip data (already masked) and optimizer
            if node_type in DATA_LOADERS:
                continue
            if node_type in OPTIMIZER_NODES:
                continue

            module = modules.get(node_id)
            if module is None:
                continue

            inputs = gather_inputs(node_id, edges, results)
            out = execute_node(node_type, module, inputs)
            if out is not None:
                results[node_id] = out


# --- Forward pass using trained modules ---

def _forward_with_trained(graph_data: dict, mask: list | None = None) -> tuple:
    """Run a forward pass using modules from _model_store (trained weights).

    Returns (modules, results, nodes, edges) — same shape as the first 4 elements
    of build_and_run_graph's return value, but with trained weights instead of random.

    Raises if trained modules don't match the current graph (missing node IDs).
    """
    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    results: dict[str, dict] = {}
    dev = get_device()

    with torch.no_grad():
        for node_id in order:
            node = nodes[node_id]
            node_type = node["type"]
            props = node.get("properties", {})

            # Data nodes: load a fresh sample (and apply mask if provided)
            loader = DATA_LOADERS.get(node_type)
            if loader:
                infer_props = {**props, "batchSize": 1}
                tensors = loader(infer_props)
                tensors = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in tensors.items()}
                if mask is not None:
                    _apply_mask_to_tensors(tensors, mask)
                results[node_id] = tensors
                continue

            # Skip optimizers
            if node_type in OPTIMIZER_NODES:
                continue

            module = trained.get(node_id)
            if module is None:
                # Trained model doesn't cover this node (graph has changed) — abort
                raise RuntimeError(f"Trained model missing module for node {node_id}")

            inputs = gather_inputs(node_id, edges, results)
            out = execute_node(node_type, module, inputs)
            if out is not None:
                results[node_id] = out

    return trained, results, nodes, edges


# --- Subgraph recursion ---

def _build_subgraph_stages(
    *,
    sg_module: SubGraphModule,
    block_name: str,
    depth: int,
    parent_path: list[str],
) -> list[dict]:
    """Walk a SubGraphModule's inner nodes and emit one stage per trainable/structural inner node.

    Handles nested subgraphs via recursive call. Skips sentinels (input/output) since they're
    redundant — the last outer stage is the subgraph's input, and the subgraph's output is
    captured by the final inner stage.
    """
    inner_nodes = sg_module.inner_nodes
    inner_edges = sg_module.inner_edges
    inner_order = sg_module.inner_order
    inner_results = getattr(sg_module, '_last_results', {})

    stages: list[dict] = []
    for inner_nid in inner_order:
        inner_node = inner_nodes[inner_nid]
        inner_type = inner_node["type"]

        # Skip sentinels (pass-through) and optimizer/loss nodes
        if inner_type in (SENTINEL_INPUT, SENTINEL_OUTPUT):
            continue
        if inner_type in OPTIMIZER_NODES:
            continue

        # Nested subgraph: recurse
        safe_key = sg_module._key_map.get(inner_nid)
        inner_mod = sg_module.inner_modules[safe_key] if safe_key and safe_key in sg_module.inner_modules else None
        if isinstance(inner_mod, SubGraphModule):
            nested_name = inner_node.get("properties", {}).get("blockName") or inner_nid
            nested_stages = _build_subgraph_stages(
                sg_module=inner_mod,
                block_name=nested_name,
                depth=depth + 1,
                parent_path=parent_path + [inner_nid],
            )
            stages.extend(nested_stages)
            continue

        # Regular inner node — emit a stage
        stage = _build_stage(
            node=inner_node,
            node_id=inner_nid,
            path=parent_path + [inner_nid],
            depth=depth,
            edges=inner_edges,
            results=inner_results,
            nodes=inner_nodes,
            module=inner_mod,
        )
        if stage:
            stage["blockName"] = block_name
            stages.append(stage)

    return stages


# --- Saliency maps (input gradient) ---

def _compute_saliency(modules: dict, nodes: dict, edges: list) -> dict | None:
    """Compute saliency = |∂logit_predicted / ∂input|, reduced across channels.

    Runs a fresh forward pass with gradient tracking on the input, then backprops
    from the predicted class logit. Returns 2D pixel data [H][W] in 0-255.
    """
    data_nid = None
    loss_nid = None
    for nid, n in nodes.items():
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid
    if not data_nid or not loss_nid:
        return None

    pred_nid = None
    for edge in edges:
        if edge["target"]["nodeId"] == loss_nid and edge["target"]["portId"] == "predictions":
            pred_nid = edge["source"]["nodeId"]
            break
    if not pred_nid:
        # For VAE and other non-classification losses, pred_nid may not exist — that's OK
        pass

    data_node = nodes[data_nid]
    loader = DATA_LOADERS.get(data_node["type"])
    if not loader:
        return None
    try:
        props = {**data_node.get("properties", {}), "batchSize": 1}
        tensors = loader(props)
    except Exception:
        return None

    dev = get_device()
    img = tensors["out"].to(dev).detach().clone()
    if img.dim() != 4:
        return None  # image datasets only
    img.requires_grad_(True)

    labels = tensors.get("labels")
    if labels is not None and isinstance(labels, torch.Tensor):
        labels = labels.to(dev)

    order = topological_sort(nodes, edges)
    results: dict[str, dict] = {data_nid: {"out": img, "labels": labels}}

    try:
        for node_id in order:
            if node_id == data_nid:
                continue
            n = nodes[node_id]
            ntype = n["type"]
            if ntype in OPTIMIZER_NODES:
                continue
            mod = modules.get(node_id)
            if mod is None:
                continue
            mod.eval()
            if ntype in ALL_LOSS_NODES:
                continue  # skip loss — we want raw logits
            inputs = gather_inputs(node_id, edges, results)
            out = execute_node(ntype, mod, inputs)
            if out is not None:
                results[node_id] = out
    except Exception:
        return None

    preds = results.get(pred_nid, {}).get("out")
    if preds is None or not isinstance(preds, torch.Tensor) or preds.dim() < 2:
        return None

    try:
        pred_class = int(preds.argmax(dim=1)[0])
        score = preds[0, pred_class]
        score.backward()
    except Exception:
        return None

    if img.grad is None:
        return None

    saliency = img.grad[0].abs()
    if saliency.dim() == 3:
        saliency = saliency.max(dim=0).values
    smin, smax = float(saliency.min()), float(saliency.max())
    rng = smax - smin if smax > smin else 1.0
    norm = ((saliency - smin) / rng * 255).clamp(0, 255).byte().cpu()

    return {
        "kind": "saliency_map",
        "pixels": norm.tolist(),
        "predictedClass": pred_class,
        "height": int(norm.shape[0]),
        "width": int(norm.shape[1]),
    }


# --- Sample extraction (for the "header" preview) ---

def _extract_sample_info(nodes: dict, results: dict) -> dict:
    """Find the data node and return a preview of its sample."""
    for node_id, node in nodes.items():
        if node["type"] in DATA_LOADERS:
            tensors = results.get(node_id, {})
            out = tensors.get("out")
            labels = tensors.get("labels")
            info: dict = {
                "datasetType": node["type"],
                "actualLabel": int(labels[0].item()) if labels is not None and isinstance(labels, torch.Tensor) else None,
            }
            # Image datasets: provide pixel preview
            if out is not None and isinstance(out, torch.Tensor) and out.dim() == 4:
                info.update(_tensor_to_preview_image(out[0], node["type"]))
            # Text datasets: return raw text and token IDs
            elif out is not None and isinstance(out, torch.Tensor) and out.dim() == 2:
                info["tokenIds"] = out[0].tolist()[:64]
                if "_texts" in tensors:
                    info["sampleText"] = tensors["_texts"][0][:500]
            return info
    return {}


def _tensor_to_preview_image(img: torch.Tensor, dataset_type: str) -> dict:
    """Convert [C, H, W] to displayable pixels, denormalizing if we know how."""
    denorm = DENORMALIZERS.get(dataset_type)
    if denorm:
        img = denorm(img)
    img = (img.clamp(0, 1) * 255).byte()
    C = img.shape[0]
    if C == 1:
        return {"imagePixels": img[0].tolist(), "imageChannels": 1}
    return {"imagePixels": img.permute(1, 2, 0).tolist(), "imageChannels": C}


# --- Stage building ---

def _build_stage(
    *,
    node: dict,
    node_id: str,
    path: list[str],
    depth: int,
    edges: list,
    results: dict,
    nodes: dict,
    module=None,  # nn.Module for this node (optional, used for weight extras)
) -> dict | None:
    """Build a stage record from a single node's input/output."""
    node_type = node["type"]

    # Get the output tensor from results
    out_dict = results.get(node_id, {})
    output = out_dict.get("out") if isinstance(out_dict, dict) else None
    if output is None or not isinstance(output, torch.Tensor):
        # No output to visualize — skip (but let loss nodes through as scalars)
        if node_type not in ALL_LOSS_NODES:
            return None

    # Get input tensor from upstream via edges (first available input)
    inputs = gather_inputs(node_id, edges, results)
    input_tensor = None
    for v in inputs.values():
        if isinstance(v, torch.Tensor):
            input_tensor = v
            break

    stage: dict = {
        "stageId": "/".join(path),
        "path": path,
        "nodeId": node_id,
        "nodeType": node_type,
        "displayName": _friendly_name(node_type),
        "depth": depth,
        "inputShape": list(input_tensor.shape) if input_tensor is not None else None,
        "outputShape": list(output.shape) if output is not None else None,
    }

    # Stats (always compute for output)
    if output is not None and isinstance(output, torch.Tensor):
        stage["stats"] = compact_stats(output)

    # Per-type visualization, extras, insight — all from the unified registry
    viz_result = get_forward_viz(node_type, module, input_tensor, output, inputs, out_dict)
    if viz_result.get("viz"):
        stage["viz"] = viz_result["viz"]
    if viz_result.get("insight"):
        stage["insight"] = viz_result["insight"]
    if viz_result.get("extras"):
        stage["extras"] = viz_result["extras"]

    return stage



# Note: All viz, extras, and insight logic has been consolidated into node_viz.py.
# step_through.py now calls get_forward_viz() from the unified registry.
# To add/modify visualizations for a node type, edit FORWARD_VIZ in node_viz.py.


# --- Utility ---

def _friendly_name(node_type: str) -> str:
    """Convert 'ml.layers.conv2d' → 'Conv2d'."""
    last = node_type.split(".")[-1]
    # Title-case if snake_case, otherwise capitalize
    if "_" in last:
        parts = last.split("_")
        return "".join(p.capitalize() for p in parts)
    return last[0].upper() + last[1:]
