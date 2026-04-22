"""Unified visualization registry for step-through (forward + backward).

Each node type can register a forward viz function and/or a backward viz function.
Functions return a dict with optional keys: viz, extras, insight.

New node types get basic viz for free via the default fallbacks — add a custom
entry only when you want richer educational visuals.

Adding a new node's viz: define the function(s) and add to FORWARD_VIZ / BACKWARD_VIZ.
No other file needs to change.

Follows the same registry pattern as node_builders.py.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from graph_builder import _safe_float, LOSS_NODES, ALL_LOSS_NODES
from data_loaders import DATA_LOADERS

# ============================================================================
# Shared helpers — used by both forward and backward viz
# ============================================================================

MAX_FEATURE_MAPS = 16
MAX_FEATURE_MAP_SIZE = 32
MAX_VECTOR_ELEMENTS = 128
TOP_K_PROBS = 5


def compact_stats(tensor: torch.Tensor) -> dict:
    """Small summary of a tensor: mean, std, min, max, sparsity, histogram."""
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


def compact_stats_with_norm(tensor: torch.Tensor) -> dict:
    """Like compact_stats but also includes L2 norm (used for gradients)."""
    s = compact_stats(tensor)
    if tensor.numel() > 0:
        s["norm"] = _safe_float(float(tensor.detach().float().flatten().norm()))
    return s


def param_grad_stats(module: nn.Module | None) -> dict | None:
    """Gradient-to-weight ratio and health indicator for a trainable module."""
    if module is None:
        return None
    grads, weights = [], []
    for p in module.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().float().flatten())
            weights.append(p.detach().float().flatten())
    if not grads:
        return None
    g = torch.cat(grads)
    w = torch.cat(weights)
    g_norm = float(g.norm())
    w_norm = float(w.norm())
    ratio = g_norm / (w_norm + 1e-12)
    health = "vanishing" if g_norm < 1e-7 else ("exploding" if g_norm > 100 else "healthy")
    return {
        "gradNorm": _safe_float(g_norm),
        "weightNorm": _safe_float(w_norm),
        "ratio": _safe_float(ratio),
        "health": health,
    }


def viz_feature_maps(fmaps: torch.Tensor) -> dict:
    """Convert [C, H, W] to a grid of channel previews (0-255 normalized)."""
    C, H, W = fmaps.shape[0], fmaps.shape[1], fmaps.shape[2]
    n_maps = min(MAX_FEATURE_MAPS, C)
    target_h = min(H, MAX_FEATURE_MAP_SIZE)
    target_w = min(W, MAX_FEATURE_MAP_SIZE)
    maps_list = []
    for c in range(n_maps):
        fm = fmaps[c].detach().float()
        fmin, fmax = float(fm.min()), float(fm.max())
        rng = fmax - fmin if fmax != fmin else 1.0
        normalized = ((fm - fmin) / rng * 255).clamp(0, 255)
        if H > target_h or W > target_w:
            normalized = nn.functional.interpolate(
                normalized.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w), mode='nearest',
            ).squeeze()
        maps_list.append(normalized.byte().tolist())
    return {
        "kind": "feature_maps",
        "featureMaps": {
            "maps": maps_list, "channels": C, "showing": n_maps,
            "height": target_h, "width": target_w,
        },
    }


def viz_vector(vec: torch.Tensor) -> dict:
    """1D vector preview — truncated if very long."""
    ft = vec.detach().float().flatten()
    n = ft.numel()
    truncated = n > MAX_VECTOR_ELEMENTS
    values = ft[:MAX_VECTOR_ELEMENTS].tolist() if truncated else ft.tolist()
    return {
        "kind": "vector",
        "vector": {
            "values": [_safe_float(v) for v in values],
            "totalLength": n, "truncated": truncated,
        },
    }


def viz_probabilities(probs: torch.Tensor) -> dict:
    """Probability distribution with top-k highlights."""
    ft = probs.detach().float().flatten()
    values = [_safe_float(float(v)) for v in ft.tolist()]
    k = min(TOP_K_PROBS, ft.numel())
    top_vals, top_idx = torch.topk(ft, k)
    top_k = [{"index": int(i), "value": _safe_float(float(v))}
             for i, v in zip(top_idx.tolist(), top_vals.tolist())]
    return {"kind": "probabilities", "probabilities": {"values": values, "topK": top_k}}


def extract_conv_kernels(module) -> dict | None:
    """Conv2d kernels as a grid of small images (mean across input channels)."""
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
        kernel = weight[f].mean(dim=0)
        kmin, kmax = float(kernel.min()), float(kernel.max())
        rng = kmax - kmin if kmax != kmin else 1.0
        normalized = ((kernel - kmin) / rng * 255).clamp(0, 255).byte()
        kernels.append(normalized.tolist())
    return {
        "kind": "conv_kernels", "kernels": kernels, "showing": n_show,
        "totalFilters": out_c, "inChannels": in_c,
        "kernelHeight": kH, "kernelWidth": kW,
    }


def extract_weight_matrix(module) -> dict | None:
    """Linear weight matrix as a heatmap (aspect-preserving downsampling)."""
    weight = None
    for name, param in module.named_parameters():
        if "weight" in name:
            weight = param.detach().cpu().float()
            break
    if weight is None or weight.dim() != 2:
        return None
    actual_rows, actual_cols = weight.shape
    vmin = _safe_float(float(weight.min()))
    vmax = _safe_float(float(weight.max()))
    MAX_DIM = 96
    mat = weight
    if mat.shape[0] > MAX_DIM or mat.shape[1] > MAX_DIM:
        scale = MAX_DIM / max(mat.shape[0], mat.shape[1])
        target_rows = max(1, round(mat.shape[0] * scale))
        target_cols = max(1, round(mat.shape[1] * scale))
        mat = nn.functional.interpolate(
            mat.unsqueeze(0).unsqueeze(0),
            size=(target_rows, target_cols), mode='area',
        ).squeeze()
        if mat.dim() == 1:
            mat = mat.unsqueeze(0)
    return {
        "kind": "weight_matrix", "data": mat.tolist(),
        "rows": mat.shape[0], "cols": mat.shape[1],
        "actualRows": actual_rows, "actualCols": actual_cols,
        "min": vmin, "max": vmax,
    }


def extract_attention_map_mha(module, inputs: dict) -> dict | None:
    """Re-run MHA with need_weights=True to capture attention weights."""
    if not hasattr(module, 'mha'):
        return None
    query = inputs.get("query")
    if query is None:
        return None
    key = inputs.get("key", query)
    value = inputs.get("value", query)
    try:
        # Apply causal mask if the wrapper has one, so the attention map
        # matches the actual attention pattern used during forward pass.
        kwargs: dict = {"need_weights": True, "average_attn_weights": True}
        if getattr(module, "is_causal", False):
            seq_len = query.shape[1]
            kwargs["attn_mask"] = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
        with torch.no_grad():
            _, attn = module.mha(query, key, value, **kwargs)
    except Exception:
        return None
    if attn is None or attn.dim() < 2:
        return None
    am = attn[0] if attn.dim() >= 3 else attn
    return _format_attention_map(am.detach().cpu().float())


def extract_attention_map_sdpa(inputs: dict) -> dict | None:
    """Compute attention weights manually for scaled_dot_product_attention."""
    query = inputs.get("query")
    if query is None:
        return None
    key = inputs.get("key", query)
    try:
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
    except Exception:
        return None
    am = attn[0] if attn.dim() >= 3 else attn
    return _format_attention_map(am.detach().cpu().float())


def _format_attention_map(am: torch.Tensor) -> dict:
    if am.dim() > 2:
        am = am[0]
    MAX = 64
    actual_rows, actual_cols = am.shape[0], am.shape[1]
    if am.shape[0] > MAX:
        am = am[:MAX]
    if am.shape[1] > MAX:
        am = am[:, :MAX]
    return {
        "kind": "attention_map", "data": am.tolist(),
        "rows": am.shape[0], "cols": am.shape[1],
        "actualRows": actual_rows, "actualCols": actual_cols,
    }


def extract_recurrent_state(out_dict: dict) -> list:
    """Extract hidden (and cell for LSTM) state from LSTM/GRU/RNN output dict."""
    extras = []
    for key, label in [("hidden", "Hidden State"), ("cell", "Cell State")]:
        t = out_dict.get(key) if isinstance(out_dict, dict) else None
        if t is None or not isinstance(t, torch.Tensor):
            continue
        if t.dim() == 3:
            v = t[0, 0]
        elif t.dim() == 2:
            v = t[0]
        else:
            v = t.flatten()
        v = v.detach().cpu().float()
        extras.append({
            "kind": "recurrent_state", "label": label,
            "values": [_safe_float(x) for x in v.tolist()],
            "totalLength": v.numel(),
        })
    return extras


def _extract_grad_weight_matrix(module) -> dict | None:
    """Weight gradient as a heatmap (aspect-preserving downsampling)."""
    weight_grad = None
    for name, param in module.named_parameters():
        if "weight" in name and param.grad is not None:
            weight_grad = param.grad.detach().cpu().float()
            break
    if weight_grad is None or weight_grad.dim() != 2:
        return None
    actual_rows, actual_cols = weight_grad.shape
    vmin = _safe_float(float(weight_grad.min()))
    vmax = _safe_float(float(weight_grad.max()))
    MAX_DIM = 96
    mat = weight_grad
    if mat.shape[0] > MAX_DIM or mat.shape[1] > MAX_DIM:
        scale = MAX_DIM / max(mat.shape[0], mat.shape[1])
        target_rows = max(1, round(mat.shape[0] * scale))
        target_cols = max(1, round(mat.shape[1] * scale))
        mat = nn.functional.interpolate(
            mat.unsqueeze(0).unsqueeze(0),
            size=(target_rows, target_cols), mode='area',
        ).squeeze()
        if mat.dim() == 1:
            mat = mat.unsqueeze(0)
    return {
        "kind": "gradient_weight_matrix", "data": mat.tolist(),
        "rows": mat.shape[0], "cols": mat.shape[1],
        "actualRows": actual_rows, "actualCols": actual_cols,
        "min": vmin, "max": vmax,
    }


def _default_viz_for_output(output: torch.Tensor) -> dict | None:
    """Pick a viz based on output tensor shape (shape-based fallback)."""
    if output.numel() == 1:
        return {"kind": "scalar", "scalar": {"value": _safe_float(float(output.flatten()[0]))}}
    if output.dim() == 4:
        return viz_feature_maps(output[0])
    if output.dim() == 3:
        return viz_vector(output[0].flatten())
    if output.dim() == 2:
        return viz_vector(output[0])
    if output.dim() == 1:
        return viz_vector(output)
    return None


# ============================================================================
# Forward viz functions — one per node type (or group)
# Signature: (node_type, module, input_tensor, output, inputs, out_dict) -> dict
# Returns: { viz?, extras?, insight? }
# ============================================================================

def forward_viz_conv2d(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None and output.dim() == 4:
        result["viz"] = viz_feature_maps(output[0])
    if module is not None:
        kernel_data = extract_conv_kernels(module)
        if kernel_data:
            result["extras"] = [kernel_data]
    if input_tensor is not None and output is not None and output.dim() == 4:
        in_c = input_tensor.shape[1]
        out_c = output.shape[1]
        result["insight"] = f"Applied {out_c} filters × {in_c} channels to produce {out_c} feature maps"
    return result


def forward_viz_conv_transpose2d(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None and output.dim() == 4:
        result["viz"] = viz_feature_maps(output[0])
    if module is not None:
        kernel_data = extract_conv_kernels(module)
        if kernel_data:
            result["extras"] = [kernel_data]
    if output is not None and output.dim() == 4:
        result["insight"] = f"Upsampled via transposed convolution to {output.shape[-2]}×{output.shape[-1]}"
    return result


def forward_viz_linear(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if module is not None:
        wm = extract_weight_matrix(module)
        if wm:
            result["extras"] = [wm]
    if input_tensor is not None and output is not None and output.dim() >= 2:
        result["insight"] = f"Linear projection: {input_tensor.shape[-1]} → {output.shape[-1]}"
    return result


def forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict):
    """Generic activation — before/after histograms."""
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if input_tensor is not None and isinstance(input_tensor, torch.Tensor) and output is not None:
        result["extras"] = [{
            "kind": "before_after_histograms",
            "input": compact_stats(input_tensor),
            "output": compact_stats(output),
        }]
    return result


def forward_viz_relu(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    if output is not None:
        zeros = float((output.detach() == 0).sum())
        total = output.numel()
        result["insight"] = f"{zeros/total*100:.0f}% of values became zero (ReLU zeros negatives)"
    return result


def forward_viz_sigmoid(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    result["insight"] = "Mapped values to (0, 1) range via sigmoid"
    return result


def forward_viz_tanh(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    result["insight"] = "Mapped values to (-1, 1) range via tanh"
    return result


def forward_viz_softmax(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    if output is not None and output.dim() == 2:
        result["viz"] = viz_probabilities(output[0])
    result["insight"] = "Converted to class probabilities (sum = 1)"
    return result


def forward_viz_gelu(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    result["insight"] = "Smooth nonlinearity (approximate ReLU)"
    return result


def forward_viz_leaky_relu(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    result["insight"] = "Leaky ReLU (small slope for negative values)"
    return result


def forward_viz_norm(node_type, module, input_tensor, output, inputs, out_dict):
    """BatchNorm, LayerNorm, GroupNorm, InstanceNorm."""
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if input_tensor is not None and isinstance(input_tensor, torch.Tensor) and output is not None:
        result["extras"] = [{
            "kind": "before_after_histograms",
            "input": compact_stats(input_tensor),
            "output": compact_stats(output),
        }]
    insights = {
        "ml.layers.batchnorm2d": "Normalized to mean≈0, std≈1 per channel, then scaled by learned γ, β",
        "ml.layers.batchnorm1d": "Normalized to mean≈0, std≈1 per channel, then scaled by learned γ, β",
        "ml.layers.layernorm": "Normalized across features per sample",
        "ml.layers.groupnorm": "Normalized within groups of channels",
        "ml.layers.instancenorm2d": "Normalized per sample, per channel (spatial mean/var)",
    }
    result["insight"] = insights.get(node_type, "Normalization layer")
    return result


def forward_viz_pooling(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if input_tensor is not None and output is not None and output.dim() == 4:
        before = f"{input_tensor.shape[-2]}×{input_tensor.shape[-1]}"
        after = f"{output.shape[-2]}×{output.shape[-1]}"
        result["insight"] = f"Downsampled spatial dims from {before} to {after}"
    return result


def forward_viz_adaptive_avgpool(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if output is not None and output.dim() == 4:
        result["insight"] = f"Adaptive pool to fixed size {output.shape[-2]}×{output.shape[-1]}"
    return result


def forward_viz_upsample(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if input_tensor is not None and output is not None and output.dim() == 4:
        result["insight"] = f"Upsampled from {input_tensor.shape[-2]}×{input_tensor.shape[-1]} to {output.shape[-2]}×{output.shape[-1]}"
    return result


def forward_viz_flatten(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if input_tensor is not None and output is not None:
        result["insight"] = f"Flattened {list(input_tensor.shape[1:])} → {list(output.shape[1:])}"
    return result


def forward_viz_dropout(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    result["insight"] = "Randomly zeroed activations (only active during training)"
    return result


def forward_viz_embedding(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if output is not None and output.dim() >= 2:
        result["insight"] = f"Looked up embeddings of dim {output.shape[-1]} for each token"
    return result


def forward_viz_recurrent(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    extras = extract_recurrent_state(out_dict)
    if extras:
        result["extras"] = extras
    result["insight"] = "Processed sequence through recurrent cells"
    return result


def forward_viz_mha(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if module is not None:
        am = extract_attention_map_mha(module, inputs)
        if am:
            result["extras"] = [am]
    result["insight"] = "Attention mechanism: weighted sum over positions"
    return result


def forward_viz_sdpa(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    am = extract_attention_map_sdpa(inputs)
    if am:
        result["extras"] = [am]
    result["insight"] = "Attention mechanism: weighted sum over positions"
    return result


def forward_viz_structural(node_type, module, input_tensor, output, inputs, out_dict):
    """Add, Concat, etc."""
    insights = {
        "ml.structural.add": "Element-wise sum (residual / skip connection)",
        "ml.structural.concat": "Concatenated inputs along specified dimension",
        "ml.structural.permute": "Reordered tensor dimensions",
        "ml.structural.sequence_pool": "Pooled sequence to single vector per sample",
    }
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    if node_type == "ml.structural.reshape" and input_tensor is not None and output is not None:
        result["insight"] = f"Reshaped {list(input_tensor.shape)} → {list(output.shape)}"
    elif node_type in insights:
        result["insight"] = insights[node_type]
    return result


def forward_viz_loss(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None and isinstance(output, torch.Tensor):
        val = float(output.detach().flatten()[0])
        result["viz"] = {"kind": "scalar", "scalar": {"value": _safe_float(val)}}
        result["insight"] = f"Loss value: {val:.4f}"
    return result


def forward_viz_data(node_type, module, input_tensor, output, inputs, out_dict):
    result: dict = {}
    if output is not None:
        result["viz"] = _default_viz_for_output(output)
    result["insight"] = "Input sample from dataset"
    return result


def forward_viz_default(node_type, module, input_tensor, output, inputs, out_dict):
    """Fallback for unknown node types."""
    result: dict = {}
    if output is not None and isinstance(output, torch.Tensor):
        result["viz"] = _default_viz_for_output(output)
    return result


# ============================================================================
# Forward viz registry
# ============================================================================

FORWARD_VIZ: dict[str, callable] = {
    "ml.layers.conv2d": forward_viz_conv2d,
    "ml.layers.conv1d": forward_viz_conv2d,
    "ml.layers.conv_transpose2d": forward_viz_conv_transpose2d,
    "ml.layers.linear": forward_viz_linear,
    "ml.activations.relu": forward_viz_relu,
    "ml.activations.sigmoid": forward_viz_sigmoid,
    "ml.activations.tanh": forward_viz_tanh,
    "ml.activations.softmax": forward_viz_softmax,
    "ml.activations.gelu": forward_viz_gelu,
    "ml.activations.leaky_relu": forward_viz_leaky_relu,
    "ml.layers.batchnorm2d": forward_viz_norm,
    "ml.layers.batchnorm1d": forward_viz_norm,
    "ml.layers.layernorm": forward_viz_norm,
    "ml.layers.groupnorm": forward_viz_norm,
    "ml.layers.instancenorm2d": forward_viz_norm,
    "ml.layers.maxpool2d": forward_viz_pooling,
    "ml.layers.maxpool1d": forward_viz_pooling,
    "ml.layers.avgpool2d": forward_viz_pooling,
    "ml.layers.adaptive_avgpool2d": forward_viz_adaptive_avgpool,
    "ml.layers.upsample": forward_viz_upsample,
    "ml.layers.flatten": forward_viz_flatten,
    "ml.layers.dropout": forward_viz_dropout,
    "ml.layers.dropout2d": forward_viz_dropout,
    "ml.layers.embedding": forward_viz_embedding,
    "ml.layers.lstm": forward_viz_recurrent,
    "ml.layers.gru": forward_viz_recurrent,
    "ml.layers.rnn": forward_viz_recurrent,
    "ml.layers.multihead_attention": forward_viz_mha,
    "ml.layers.attention": forward_viz_sdpa,
    "ml.structural.add": forward_viz_structural,
    "ml.structural.concat": forward_viz_structural,
    "ml.structural.reshape": forward_viz_structural,
    "ml.structural.permute": forward_viz_structural,
    "ml.structural.sequence_pool": forward_viz_structural,
    "ml.structural.reparameterize": forward_viz_structural,
    "ml.preprocessing.tokenizer": forward_viz_structural,
}

for _lt in ALL_LOSS_NODES:
    FORWARD_VIZ[_lt] = forward_viz_loss
for _dt in DATA_LOADERS:
    FORWARD_VIZ[_dt] = forward_viz_data
FORWARD_VIZ["ml.gan.noise_input"] = forward_viz_data
FORWARD_VIZ["ml.diffusion.noise_scheduler"] = forward_viz_data
FORWARD_VIZ["ml.diffusion.timestep_embed"] = forward_viz_data


def get_forward_viz(node_type: str, module, input_tensor, output, inputs, out_dict) -> dict:
    """Look up the forward viz function for a node type and call it."""
    fn = FORWARD_VIZ.get(node_type, forward_viz_default)
    return fn(node_type, module, input_tensor, output, inputs, out_dict)


# ============================================================================
# Backward viz functions — one per node type (or group)
# Signature: (module, activation, gradient) -> dict
# Returns: { viz?, extras?, insight? }
# ============================================================================

def backward_viz_conv2d(module, activation, gradient):
    result: dict = {"extras": []}
    if gradient is not None and gradient.dim() == 4:
        result["viz"] = viz_feature_maps(gradient[0].abs())

    weight_grad = None
    for name, param in module.named_parameters():
        if "weight" in name and param.grad is not None:
            weight_grad = param.grad.detach().cpu().float()
            break
    if weight_grad is not None and weight_grad.dim() == 4:
        out_c, in_c, kH, kW = weight_grad.shape
        n_show = min(32, out_c)
        kernels = []
        for f in range(n_show):
            kernel = weight_grad[f].mean(dim=0).abs()
            kmin, kmax = float(kernel.min()), float(kernel.max())
            rng = kmax - kmin if kmax != kmin else 1.0
            normalized = ((kernel - kmin) / rng * 255).clamp(0, 255).byte()
            kernels.append(normalized.tolist())
        result["extras"].append({
            "kind": "gradient_kernels", "kernels": kernels, "showing": n_show,
            "totalFilters": out_c, "kernelHeight": kH, "kernelWidth": kW,
        })
        # Spatial gradient heatmap
        if gradient is not None and gradient.dim() == 4:
            spatial = gradient[0].abs().mean(dim=0).float()
            H, W = spatial.shape
            smin, smax = float(spatial.min()), float(spatial.max())
            rng = smax - smin if smax > smin else 1.0
            norm = ((spatial - smin) / rng * 255).clamp(0, 255).byte().cpu()
            result["extras"].append({
                "kind": "spatial_gradient_heatmap",
                "pixels": norm.tolist(), "height": H, "width": W,
            })
        # Per-filter bars
        per_filter = [_safe_float(float(weight_grad[f].norm())) for f in range(weight_grad.shape[0])]
        result["extras"].append({
            "kind": "per_unit_gradient_bars",
            "values": per_filter, "label": "Per-Filter Gradient Norm",
        })
        norms = [float(weight_grad[f].norm()) for f in range(weight_grad.shape[0])]
        max_idx = norms.index(max(norms))
        result["insight"] = f"Filter {max_idx} has the largest gradient ({norms[max_idx]:.4f}) — it's learning the most from this sample"

    if not result["extras"]:
        del result["extras"]
    return result


def backward_viz_linear(module, activation, gradient):
    result: dict = {"extras": []}
    if gradient is not None:
        per_neuron = gradient[0].abs() if gradient.dim() == 2 else gradient.abs().flatten()
        result["viz"] = viz_vector(per_neuron)
    gm = _extract_grad_weight_matrix(module)
    if gm:
        result["extras"].append(gm)
    if gradient is not None and gradient.dim() >= 2:
        flat = gradient[0].abs().flatten() if gradient.dim() >= 2 else gradient.abs().flatten()
        max_idx = int(flat.argmax())
        max_val = float(flat[max_idx])
        result["insight"] = f"Neuron {max_idx} has the largest gradient ({max_val:.4f}) — the loss is most sensitive to its output"
    if not result["extras"]:
        del result["extras"]
    return result


def _backward_viz_generic_activation(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        result["viz"] = _default_viz_for_output(gradient) if gradient.dim() <= 2 else (
            viz_feature_maps(gradient[0].abs()) if gradient.dim() == 4 else viz_vector(gradient.flatten())
        )
    if gradient is not None and activation is not None:
        result["extras"] = [{
            "kind": "before_after_histograms",
            "input": compact_stats_with_norm(activation),
            "output": compact_stats_with_norm(gradient),
        }]
    return result


def backward_viz_relu(module, activation, gradient):
    result = _backward_viz_generic_activation(module, activation, gradient)
    if gradient is not None:
        dead = float((gradient.detach() == 0).sum())
        total = gradient.numel()
        result["insight"] = f"{dead/total*100:.0f}% of gradients were killed (zeroed by ReLU's dead zone)"
    return result


def backward_viz_sigmoid(module, activation, gradient):
    result = _backward_viz_generic_activation(module, activation, gradient)
    if gradient is not None:
        g = gradient.detach().float()
        result["insight"] = f"Sigmoid compressed gradient range to [{float(g.min()):.4f}, {float(g.max()):.4f}]"
    return result


def backward_viz_tanh(module, activation, gradient):
    result = _backward_viz_generic_activation(module, activation, gradient)
    if gradient is not None:
        g = gradient.detach().float()
        result["insight"] = f"Tanh compressed gradient range to [{float(g.min()):.4f}, {float(g.max()):.4f}]"
    return result


def backward_viz_batchnorm(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        result["viz"] = viz_feature_maps(gradient[0].abs()) if gradient.dim() == 4 else viz_vector(gradient.flatten())
    extras = []
    gamma_grad = None
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if "weight" in name:
            gamma_grad = param.grad.detach().cpu().float()
    if gamma_grad is not None:
        extras.append({
            "kind": "per_unit_gradient_bars",
            "values": [_safe_float(float(v)) for v in gamma_grad.tolist()],
            "label": "Scale (γ) Gradient per Channel",
        })
    if extras:
        result["extras"] = extras
    if gradient is not None and activation is not None:
        g_std = float(gradient.detach().float().std())
        a_std = float(activation.detach().float().std())
        result["insight"] = f"BatchNorm redistributed gradients — input std {a_std:.4f}, gradient std {g_std:.4f}"
    return result


def backward_viz_maxpool(module, activation, gradient):
    result: dict = {}
    if gradient is not None and gradient.dim() == 4:
        result["viz"] = viz_feature_maps(gradient[0].abs())
        spatial = gradient[0].abs().mean(dim=0).float()
        H, W = spatial.shape
        smin, smax = float(spatial.min()), float(spatial.max())
        rng = smax - smin if smax > smin else 1.0
        norm = ((spatial - smin) / rng * 255).clamp(0, 255).byte().cpu()
        result["extras"] = [{"kind": "spatial_gradient_heatmap", "pixels": norm.tolist(), "height": H, "width": W}]
    result["insight"] = "Gradients only flow through max-selected positions — other positions receive zero gradient"
    return result


def backward_viz_avgpool(module, activation, gradient):
    result: dict = {}
    if gradient is not None and gradient.dim() == 4:
        result["viz"] = viz_feature_maps(gradient[0].abs())
        spatial = gradient[0].abs().mean(dim=0).float()
        H, W = spatial.shape
        smin, smax = float(spatial.min()), float(spatial.max())
        rng = smax - smin if smax > smin else 1.0
        norm = ((spatial - smin) / rng * 255).clamp(0, 255).byte().cpu()
        result["extras"] = [{"kind": "spatial_gradient_heatmap", "pixels": norm.tolist(), "height": H, "width": W}]
    result["insight"] = "Gradients distributed equally across the pooling window"
    return result


def backward_viz_lstm(module, activation, gradient):
    result: dict = {}
    if gradient is not None and gradient.dim() == 3:
        per_step = gradient[0].norm(dim=1).detach().float()
        values = [_safe_float(float(v)) for v in per_step.tolist()]
        result["viz"] = {"kind": "vector", "vector": {"values": values, "totalLength": len(values), "truncated": False}}
        if len(values) > 1:
            ratio = float(per_step[0]) / (float(per_step[-1]) + 1e-12)
            if ratio < 0.1:
                result["insight"] = f"Gradients at early timesteps are {ratio:.1f}× smaller — significant vanishing gradient"
            elif ratio < 0.5:
                result["insight"] = f"Gradients at early timesteps are {ratio:.1f}× smaller — mild vanishing gradient"
            else:
                result["insight"] = f"Gradient ratio (first/last timestep): {ratio:.2f} — gradients flow well through time"
    return result


def backward_viz_embedding(module, activation, gradient):
    result: dict = {}
    if gradient is not None and gradient.dim() >= 2:
        g = gradient[0] if gradient.dim() == 3 else gradient
        per_token = g.norm(dim=-1).detach().float() if g.dim() == 2 else g.abs().float()
        values = [_safe_float(float(v)) for v in per_token.tolist()]
        result["viz"] = {"kind": "vector", "vector": {"values": values, "totalLength": len(values), "truncated": False}}
        if values:
            max_idx = values.index(max(values))
            result["insight"] = f"Token at position {max_idx} has the largest gradient — it matters most for the prediction"
    return result


def backward_viz_attention(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        result["viz"] = viz_vector(gradient.flatten())
    result["insight"] = "Gradient of attention output — shows which attended positions affected the loss most"
    return result


def backward_viz_loss(module, activation, gradient):
    result: dict = {}
    if activation is not None and isinstance(activation, torch.Tensor):
        val = _safe_float(float(activation.detach().flatten()[0]))
        result["viz"] = {"kind": "scalar", "scalar": {"value": val}}
    result["insight"] = "Backward pass starts here. Gradient of loss w.r.t. itself = 1.0"
    return result


def backward_viz_default(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = viz_feature_maps(gradient[0].abs())
        elif gradient.dim() >= 1:
            result["viz"] = viz_vector(gradient.flatten())
        result["insight"] = f"Gradient norm: {float(gradient.detach().float().norm()):.6f}"
    return result


def _backward_viz_structural(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = viz_feature_maps(gradient[0].abs())
        else:
            result["viz"] = viz_vector(gradient.flatten())
    return result


def _backward_viz_flatten(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        result["viz"] = viz_vector(gradient.flatten())
    if activation is not None and gradient is not None:
        result["insight"] = f"Gradient reshaped from {list(gradient.shape)} back to spatial layout upstream"
    return result


def _backward_viz_dropout(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = viz_feature_maps(gradient[0].abs())
        else:
            result["viz"] = viz_vector(gradient.flatten())
    result["insight"] = "Dropout mask from forward pass determines which gradients are zeroed"
    return result


# ============================================================================
# Backward viz registry
# ============================================================================

BACKWARD_VIZ: dict[str, callable] = {
    "ml.layers.conv2d": backward_viz_conv2d,
    "ml.layers.conv1d": backward_viz_conv2d,
    "ml.layers.conv_transpose2d": backward_viz_conv2d,
    "ml.layers.linear": backward_viz_linear,
    "ml.activations.relu": backward_viz_relu,
    "ml.activations.sigmoid": backward_viz_sigmoid,
    "ml.activations.tanh": backward_viz_tanh,
    "ml.activations.gelu": _backward_viz_generic_activation,
    "ml.activations.leaky_relu": _backward_viz_generic_activation,
    "ml.activations.softmax": _backward_viz_generic_activation,
    "ml.layers.batchnorm2d": backward_viz_batchnorm,
    "ml.layers.batchnorm1d": backward_viz_batchnorm,
    "ml.layers.layernorm": backward_viz_batchnorm,
    "ml.layers.groupnorm": backward_viz_batchnorm,
    "ml.layers.instancenorm2d": backward_viz_batchnorm,
    "ml.layers.maxpool2d": backward_viz_maxpool,
    "ml.layers.avgpool2d": backward_viz_avgpool,
    "ml.layers.adaptive_avgpool2d": backward_viz_avgpool,
    "ml.layers.upsample": backward_viz_default,
    "ml.layers.lstm": backward_viz_lstm,
    "ml.layers.gru": backward_viz_lstm,
    "ml.layers.rnn": backward_viz_lstm,
    "ml.layers.embedding": backward_viz_embedding,
    "ml.layers.attention": backward_viz_attention,
    "ml.layers.multihead_attention": backward_viz_attention,
    "ml.layers.flatten": _backward_viz_flatten,
    "ml.layers.dropout": _backward_viz_dropout,
    "ml.layers.dropout2d": _backward_viz_dropout,
    "ml.structural.add": _backward_viz_structural,
    "ml.structural.concat": _backward_viz_structural,
    "ml.structural.reshape": _backward_viz_structural,
    "ml.structural.permute": _backward_viz_structural,
    "ml.structural.sequence_pool": _backward_viz_structural,
    "ml.structural.reparameterize": _backward_viz_structural,
}

for _lt in ALL_LOSS_NODES:
    BACKWARD_VIZ[_lt] = backward_viz_loss


def get_backward_viz(node_type: str, module, activation, gradient) -> dict:
    """Look up the backward viz function for a node type and call it."""
    fn = BACKWARD_VIZ.get(node_type, backward_viz_default)
    return fn(module, activation, gradient)
