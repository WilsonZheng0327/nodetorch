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
    OPTIMIZER_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    SubGraphModule,
)
from forward_utils import execute_node
from data_loaders import DATA_LOADERS, DENORMALIZERS


# --- Visualization kind constants (discriminator for Stage.viz) ---

VIZ_IMAGE = "image"
VIZ_FEATURE_MAPS = "feature_maps"
VIZ_VECTOR = "vector"
VIZ_PROBABILITIES = "probabilities"
VIZ_SCALAR = "scalar"

# Display limits (keep response size reasonable)
MAX_FEATURE_MAPS = 16
MAX_FEATURE_MAP_SIZE = 32
MAX_VECTOR_ELEMENTS = 128
TOP_K_PROBS = 5


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
        if n["type"] in LOSS_NODES:
            loss_nid = nid
    if not data_nid or not loss_nid:
        return None

    pred_nid = None
    for edge in edges:
        if edge["target"]["nodeId"] == loss_nid and edge["target"]["portId"] == "predictions":
            pred_nid = edge["source"]["nodeId"]
            break
    if not pred_nid:
        return None

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
            if ntype in LOSS_NODES:
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
        if node_type not in LOSS_NODES:
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
        stage["stats"] = _compact_stats(output)

    # Per-type visualization
    viz = _build_viz(node_type, input_tensor, output)
    if viz:
        stage["viz"] = viz

    # Per-type insight
    insight = _build_insight(node_type, input_tensor, output)
    if insight:
        stage["insight"] = insight

    # Extras — additional type-specific visualizations
    extras = _build_extras(node_type, input_tensor, output, module, inputs, out_dict)
    if extras:
        stage["extras"] = extras

    return stage


# --- Stats helpers ---

def _compact_stats(tensor: torch.Tensor) -> dict:
    """Small summary of a tensor (used for every stage)."""
    if tensor.numel() == 0:
        return {}
    ft = tensor.detach().float().flatten()
    stats: dict = {
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
        "min": _safe_float(float(ft.min())),
        "max": _safe_float(float(ft.max())),
    }
    if ft.numel() > 1:
        stats["sparsity"] = _safe_float(float((ft == 0).sum()) / ft.numel())
        hist = torch.histogram(ft.cpu(), bins=20)
        stats["histBins"] = [_safe_float(float(x)) for x in hist.bin_edges[:-1]]
        stats["histCounts"] = [int(x) for x in hist.hist]
    return stats


# --- Viz builder (discriminated by node_type and tensor shape) ---

def _build_viz(node_type: str, input_tensor: torch.Tensor | None, output: torch.Tensor | None) -> dict | None:
    """Choose an appropriate visualization for this node's output tensor."""
    if output is None or not isinstance(output, torch.Tensor):
        return None

    # Loss nodes → scalar
    if node_type in LOSS_NODES or output.numel() == 1:
        return {"kind": VIZ_SCALAR, "scalar": {"value": _safe_float(float(output.flatten()[0]))}}

    # Softmax / final layer probabilities → probability bars
    # Detect: 2D output [B, num_classes] from softmax or the last linear before a loss
    if node_type == "ml.activations.softmax":
        if output.dim() == 2:
            return _viz_probabilities(output[0])

    # Conv/pooling/batchnorm/etc. with 4D output → feature maps
    if output.dim() == 4:
        return _viz_feature_maps(output[0])

    # 3D output [B, seq, H] — show as heat-ish matrix (skip for now, falls through to vector)
    if output.dim() == 3:
        # Flatten to vector — shows the sequence
        return _viz_vector(output[0].flatten())

    # 2D output [B, features] → vector
    if output.dim() == 2:
        return _viz_vector(output[0])

    # 1D output → vector
    if output.dim() == 1:
        return _viz_vector(output)

    return None


def _viz_feature_maps(fmaps: torch.Tensor) -> dict:
    """Convert [C, H, W] to a grid of channel previews."""
    C = fmaps.shape[0]
    n_maps = min(MAX_FEATURE_MAPS, C)
    H, W = fmaps.shape[1], fmaps.shape[2]

    # Downsample spatially if too large
    target_h = min(H, MAX_FEATURE_MAP_SIZE)
    target_w = min(W, MAX_FEATURE_MAP_SIZE)

    maps_list = []
    for c in range(n_maps):
        fm = fmaps[c].detach().float()
        # Normalize to 0-255 per channel
        fmin, fmax = float(fm.min()), float(fm.max())
        rng = fmax - fmin if fmax != fmin else 1.0
        normalized = ((fm - fmin) / rng * 255).clamp(0, 255)
        if H > target_h or W > target_w:
            normalized = torch.nn.functional.interpolate(
                normalized.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w),
                mode='nearest',
            ).squeeze()
        maps_list.append(normalized.byte().tolist())

    return {
        "kind": VIZ_FEATURE_MAPS,
        "featureMaps": {
            "maps": maps_list,
            "channels": C,
            "showing": n_maps,
            "height": target_h,
            "width": target_w,
        },
    }


def _viz_vector(vec: torch.Tensor) -> dict:
    """1D vector preview — truncated if very long."""
    ft = vec.detach().float()
    n = ft.numel()
    if n > MAX_VECTOR_ELEMENTS:
        # Take first, last, and stride through middle
        values = ft[:MAX_VECTOR_ELEMENTS].tolist()
        truncated = True
    else:
        values = ft.tolist()
        truncated = False
    return {
        "kind": VIZ_VECTOR,
        "vector": {
            "values": [_safe_float(v) for v in values],
            "totalLength": n,
            "truncated": truncated,
        },
    }


def _viz_probabilities(probs: torch.Tensor) -> dict:
    """Show probability distribution with top-k highlights."""
    ft = probs.detach().float().flatten()
    values = [_safe_float(float(v)) for v in ft.tolist()]
    # Top-k with indices
    k = min(TOP_K_PROBS, ft.numel())
    top_vals, top_idx = torch.topk(ft, k)
    top_k = [
        {"index": int(i), "value": _safe_float(float(v))}
        for i, v in zip(top_idx.tolist(), top_vals.tolist())
    ]
    return {
        "kind": VIZ_PROBABILITIES,
        "probabilities": {"values": values, "topK": top_k},
    }


# --- Extras: additional type-specific visualizations ---
#
# Extras are optional per-stage extensions that render as additional panels in the
# detail view. Each extra has a "kind" discriminator so new ones can be added without
# breaking existing frontend code.

def _build_extras(node_type: str, input_tensor, output, module, inputs: dict, out_dict: dict) -> list | None:
    """Return a list of extras for this node, or None."""
    extras: list = []

    # Before/after histograms: most informative for normalization/activation layers
    if node_type in (
        "ml.activations.relu",
        "ml.activations.sigmoid",
        "ml.activations.tanh",
        "ml.activations.gelu",
        "ml.activations.leaky_relu",
        "ml.activations.softmax",
        "ml.layers.batchnorm2d",
        "ml.layers.batchnorm1d",
        "ml.layers.layernorm",
        "ml.layers.groupnorm",
        "ml.layers.instancenorm2d",
    ):
        if (input_tensor is not None and isinstance(input_tensor, torch.Tensor)
                and output is not None and isinstance(output, torch.Tensor)):
            extras.append({
                "kind": "before_after_histograms",
                "input": _compact_stats(input_tensor),
                "output": _compact_stats(output),
            })

    # Conv2d learned kernels
    if node_type == "ml.layers.conv2d" and module is not None:
        kernel_data = _extract_conv_kernels(module)
        if kernel_data:
            extras.append(kernel_data)

    # Linear weight matrix
    if node_type == "ml.layers.linear" and module is not None:
        wm_data = _extract_weight_matrix(module)
        if wm_data:
            extras.append(wm_data)

    # Attention map (MHA + scaled-dot-product Attention)
    if node_type == "ml.layers.multihead_attention" and module is not None:
        am = _extract_attention_map_mha(module, inputs)
        if am:
            extras.append(am)
    elif node_type == "ml.layers.attention":
        am = _extract_attention_map_sdpa(inputs)
        if am:
            extras.append(am)

    # Recurrent state (LSTM/GRU/RNN return dicts with hidden/cell)
    if node_type in ("ml.layers.lstm", "ml.layers.gru", "ml.layers.rnn"):
        rs = _extract_recurrent_state(out_dict)
        extras.extend(rs)

    return extras if extras else None


def _extract_attention_map_mha(module, inputs: dict) -> dict | None:
    """Re-run MHA with need_weights=True to capture attention weights."""
    if not hasattr(module, 'mha'):
        return None
    query = inputs.get("query")
    key = inputs.get("key")
    value = inputs.get("value")
    if query is None:
        return None
    # Default to self-attention if key/value not provided
    key = key if key is not None else query
    value = value if value is not None else query
    try:
        with torch.no_grad():
            _, attn = module.mha(query, key, value, need_weights=True, average_attn_weights=True)
    except Exception:
        return None
    if attn is None or attn.dim() < 2:
        return None
    am = attn[0] if attn.dim() >= 3 else attn  # take first sample
    am = am.detach().cpu().float()
    return _format_attention_map(am)


def _extract_attention_map_sdpa(inputs: dict) -> dict | None:
    """Compute attention weights manually for scaled_dot_product_attention."""
    import math
    query = inputs.get("query")
    key = inputs.get("key")
    if query is None:
        return None
    key = key if key is not None else query
    try:
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
    except Exception:
        return None
    am = attn[0] if attn.dim() >= 3 else attn
    am = am.detach().cpu().float()
    return _format_attention_map(am)


def _format_attention_map(am: torch.Tensor) -> dict:
    """Downsample an attention matrix if large and return the extras dict."""
    if am.dim() > 2:
        am = am[0]  # first head, if still 3D
    MAX = 64
    actual_rows, actual_cols = am.shape[0], am.shape[1]
    if am.shape[0] > MAX:
        am = am[:MAX]
    if am.shape[1] > MAX:
        am = am[:, :MAX]
    return {
        "kind": "attention_map",
        "data": am.tolist(),
        "rows": am.shape[0],
        "cols": am.shape[1],
        "actualRows": actual_rows,
        "actualCols": actual_cols,
    }


def _extract_recurrent_state(out_dict: dict) -> list:
    """Extract hidden (and cell for LSTM) state from LSTM/GRU/RNN output dict."""
    extras = []
    for key, label in [("hidden", "Hidden State"), ("cell", "Cell State")]:
        t = out_dict.get(key) if isinstance(out_dict, dict) else None
        if t is None or not isinstance(t, torch.Tensor):
            continue
        # [num_layers * D, B, H] → first layer/direction, first sample, all hidden units
        if t.dim() == 3:
            v = t[0, 0]
        elif t.dim() == 2:
            v = t[0]
        else:
            v = t.flatten()
        v = v.detach().cpu().float()
        extras.append({
            "kind": "recurrent_state",
            "label": label,
            "values": [_safe_float(x) for x in v.tolist()],
            "totalLength": v.numel(),
        })
    return extras


def _extract_conv_kernels(module) -> dict | None:
    """Extract Conv2d kernels as a grid of small images (mean across input channels)."""
    weight = None
    for name, param in module.named_parameters():
        if "weight" in name:
            weight = param.detach().cpu().float()
            break
    if weight is None or weight.dim() != 4:
        return None

    out_c, in_c, kH, kW = weight.shape
    n_show = min(32, out_c)
    kernels = []
    for f in range(n_show):
        # Average across input channels to get one kH×kW image per filter
        kernel = weight[f].mean(dim=0)
        kmin, kmax = float(kernel.min()), float(kernel.max())
        rng = kmax - kmin if kmax != kmin else 1.0
        normalized = ((kernel - kmin) / rng * 255).clamp(0, 255).byte()
        kernels.append(normalized.tolist())

    return {
        "kind": "conv_kernels",
        "kernels": kernels,
        "showing": n_show,
        "totalFilters": out_c,
        "inChannels": in_c,
        "kernelHeight": kH,
        "kernelWidth": kW,
    }


def _extract_weight_matrix(module) -> dict | None:
    """Extract Linear weight matrix as a heatmap (downsampled if large)."""
    weight = None
    for name, param in module.named_parameters():
        if "weight" in name:
            weight = param.detach().cpu().float()
            break
    if weight is None or weight.dim() != 2:
        return None

    actual_rows, actual_cols = weight.shape[0], weight.shape[1]
    vmin = _safe_float(float(weight.min()))
    vmax = _safe_float(float(weight.max()))

    # Downsample with area averaging if too large, preserving aspect ratio
    MAX_DIM = 96
    mat = weight
    if mat.shape[0] > MAX_DIM or mat.shape[1] > MAX_DIM:
        scale = MAX_DIM / max(mat.shape[0], mat.shape[1])
        target_rows = max(1, round(mat.shape[0] * scale))
        target_cols = max(1, round(mat.shape[1] * scale))
        mat = torch.nn.functional.interpolate(
            mat.unsqueeze(0).unsqueeze(0),
            size=(target_rows, target_cols),
            mode='area',
        ).squeeze()
        if mat.dim() == 1:
            mat = mat.unsqueeze(0)

    return {
        "kind": "weight_matrix",
        "data": mat.tolist(),
        "rows": mat.shape[0],
        "cols": mat.shape[1],
        "actualRows": actual_rows,
        "actualCols": actual_cols,
        "min": vmin,
        "max": vmax,
    }


# --- Insight generators (plain-English explanations) ---

def _build_insight(node_type: str, input_tensor: torch.Tensor | None, output: torch.Tensor | None) -> str | None:
    """Generate a short sentence explaining what this layer did."""
    if output is None or not isinstance(output, torch.Tensor):
        return None

    if node_type == "ml.activations.relu":
        zeros = float((output.detach() == 0).sum())
        total = output.numel()
        return f"{zeros/total*100:.0f}% of values became zero (ReLU zeros negatives)"

    if node_type == "ml.activations.sigmoid":
        return "Mapped values to (0, 1) range via sigmoid"

    if node_type == "ml.activations.tanh":
        return "Mapped values to (-1, 1) range via tanh"

    if node_type == "ml.activations.softmax":
        return "Converted to class probabilities (sum = 1)"

    if node_type == "ml.activations.gelu":
        return "Smooth nonlinearity (approximate ReLU)"

    if node_type == "ml.activations.leaky_relu":
        return "Leaky ReLU (small slope for negative values)"

    if node_type == "ml.layers.conv2d":
        if input_tensor is not None and output.dim() == 4:
            in_c = input_tensor.shape[1]
            out_c = output.shape[1]
            return f"Applied {out_c} filters × {in_c} channels to produce {out_c} feature maps"

    if node_type == "ml.layers.conv_transpose2d":
        if output.dim() == 4:
            return f"Upsampled via transposed convolution to {output.shape[-2]}×{output.shape[-1]}"

    if node_type in ("ml.layers.maxpool2d", "ml.layers.avgpool2d"):
        if input_tensor is not None and output.dim() == 4:
            before = f"{input_tensor.shape[-2]}×{input_tensor.shape[-1]}"
            after = f"{output.shape[-2]}×{output.shape[-1]}"
            return f"Downsampled spatial dims from {before} to {after}"

    if node_type == "ml.layers.adaptive_avgpool2d":
        if output.dim() == 4:
            return f"Adaptive pool to fixed size {output.shape[-2]}×{output.shape[-1]}"

    if node_type == "ml.layers.upsample":
        if input_tensor is not None and output.dim() == 4:
            return f"Upsampled from {input_tensor.shape[-2]}×{input_tensor.shape[-1]} to {output.shape[-2]}×{output.shape[-1]}"

    if node_type in ("ml.layers.batchnorm2d", "ml.layers.batchnorm1d"):
        return "Normalized to mean≈0, std≈1 per channel, then scaled by learned γ, β"

    if node_type == "ml.layers.layernorm":
        return "Normalized across features per sample"

    if node_type == "ml.layers.groupnorm":
        return "Normalized within groups of channels"

    if node_type == "ml.layers.instancenorm2d":
        return "Normalized per sample, per channel (spatial mean/var)"

    if node_type == "ml.layers.dropout" or node_type == "ml.layers.dropout2d":
        return "Randomly zeroed activations (only active during training)"

    if node_type == "ml.layers.flatten":
        if input_tensor is not None:
            return f"Flattened {list(input_tensor.shape[1:])} → {list(output.shape[1:])}"

    if node_type == "ml.layers.linear":
        if input_tensor is not None and output.dim() >= 2:
            return f"Linear projection: {input_tensor.shape[-1]} → {output.shape[-1]}"

    if node_type == "ml.layers.embedding":
        if output.dim() >= 2:
            return f"Looked up embeddings of dim {output.shape[-1]} for each token"

    if node_type in ("ml.layers.lstm", "ml.layers.gru", "ml.layers.rnn"):
        return "Processed sequence through recurrent cells"

    if node_type in ("ml.layers.attention", "ml.layers.multihead_attention"):
        return "Attention mechanism: weighted sum over positions"

    if node_type == "ml.structural.add":
        return "Element-wise sum (residual / skip connection)"

    if node_type == "ml.structural.concat":
        return "Concatenated inputs along specified dimension"

    if node_type == "ml.structural.reshape":
        if input_tensor is not None:
            return f"Reshaped {list(input_tensor.shape)} → {list(output.shape)}"

    if node_type == "ml.structural.permute":
        return "Reordered tensor dimensions"

    if node_type == "ml.structural.sequence_pool":
        return "Pooled sequence to single vector per sample"

    if node_type in LOSS_NODES:
        val = float(output.detach().flatten()[0])
        return f"Loss value: {val:.4f}"

    if node_type in DATA_LOADERS:
        return "Input sample from dataset"

    return None


# --- Utility ---

def _friendly_name(node_type: str) -> str:
    """Convert 'ml.layers.conv2d' → 'Conv2d'."""
    last = node_type.split(".")[-1]
    # Title-case if snake_case, otherwise capitalize
    if "_" in last:
        parts = last.split("_")
        return "".join(p.capitalize() for p in parts)
    return last[0].upper() + last[1:]
