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
    _safe_float,
    LOSS_NODES,
    OPTIMIZER_NODES,
    SUBGRAPH_TYPE,
    SENTINEL_INPUT,
    SENTINEL_OUTPUT,
    SubGraphModule,
)
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

def run_step_through(graph_data: dict, sample_idx: int | None = None) -> dict:
    """Run a forward pass and return ordered stages.

    Args:
        graph_data: serialized NodeTorch graph (same format as /forward)
        sample_idx: optional index into the dataset (not used in Phase 1; random batch used)

    Returns:
        { "stages": [...], "sample": { "imagePixels": ..., "imageChannels": ..., "actualLabel": ... } }
    """
    # Deep-copy so we can override batch size without mutating caller's data
    graph_data = copy.deepcopy(graph_data)

    # Override batch size to 1 for step-through (showing one sample at a time)
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in DATA_LOADERS:
            n["properties"] = {**n.get("properties", {}), "batchSize": 1}

    # Run the graph — build modules and collect per-node outputs in `results`
    with torch.no_grad():
        modules, results, _, nodes, edges = build_and_run_graph(graph_data)

    # Topological order for stage sequencing
    from graph_builder import topological_sort
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
        )
        if stage:
            stages.append(stage)

    return {"stages": stages, "sample": sample_info}


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
        inner_mod = sg_module.inner_modules.get(safe_key) if safe_key else None
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
        )
        if stage:
            stage["blockName"] = block_name
            stages.append(stage)

    return stages


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
                "actualLabel": int(labels[0]) if labels is not None else None,
            }
            # Image datasets: provide pixel preview
            if out is not None and isinstance(out, torch.Tensor) and out.dim() == 4:
                info.update(_tensor_to_preview_image(out[0], node["type"]))
            # Text datasets: just return the token IDs (not very useful, but at least shape info)
            elif out is not None and isinstance(out, torch.Tensor) and out.dim() == 2:
                info["tokenIds"] = out[0].tolist()[:64]  # first 64 tokens
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
