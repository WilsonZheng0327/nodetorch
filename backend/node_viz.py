"""Visualization registry for backward (gradient) step-through.

Each function takes (module, activation, gradient) and returns a dict with
optional keys: viz, extras, insight, paramGradStats.

New node types get basic viz for free via backward_viz_default — add a custom
entry only when you want richer educational visuals.

Follows the same registry pattern as node_builders.py.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from graph_builder import _safe_float, LOSS_NODES

# --- Limits ---
MAX_FEATURE_MAPS = 16
MAX_FEATURE_MAP_SIZE = 32
MAX_VECTOR_ELEMENTS = 128

# --- Shared helpers ---

def _compact_grad_stats(tensor: torch.Tensor) -> dict:
    """Summary stats for a gradient tensor."""
    if tensor.numel() == 0:
        return {}
    ft = tensor.detach().float().flatten()
    stats: dict = {
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
        "min": _safe_float(float(ft.min())),
        "max": _safe_float(float(ft.max())),
        "norm": _safe_float(float(ft.norm())),
    }
    if ft.numel() > 1:
        stats["sparsity"] = _safe_float(float((ft == 0).sum()) / ft.numel())
        hist = torch.histogram(ft.cpu(), bins=20)
        stats["histBins"] = [_safe_float(float(x)) for x in hist.bin_edges[:-1]]
        stats["histCounts"] = [int(x) for x in hist.hist]
    return stats


def _param_grad_stats(module: nn.Module | None) -> dict | None:
    """Compute gradient-to-weight ratio and health for a trainable module."""
    if module is None:
        return None
    grads = []
    weights = []
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
    if g_norm < 1e-7:
        health = "vanishing"
    elif g_norm > 100:
        health = "exploding"
    else:
        health = "healthy"
    return {
        "gradNorm": _safe_float(g_norm),
        "weightNorm": _safe_float(w_norm),
        "ratio": _safe_float(ratio),
        "health": health,
    }


def _grad_feature_maps(grad: torch.Tensor) -> dict | None:
    """Render gradient magnitude as feature maps (for 4D gradient [B,C,H,W])."""
    if grad.dim() != 4:
        return None
    g = grad[0].abs()  # [C, H, W]
    C, H, W = g.shape
    n_maps = min(MAX_FEATURE_MAPS, C)
    target_h = min(H, MAX_FEATURE_MAP_SIZE)
    target_w = min(W, MAX_FEATURE_MAP_SIZE)
    maps_list = []
    for c in range(n_maps):
        fm = g[c].float()
        fmin, fmax = float(fm.min()), float(fm.max())
        rng = fmax - fmin if fmax != fmin else 1.0
        normalized = ((fm - fmin) / rng * 255).clamp(0, 255)
        if H > target_h or W > target_w:
            normalized = torch.nn.functional.interpolate(
                normalized.unsqueeze(0).unsqueeze(0),
                size=(target_h, target_w), mode='nearest',
            ).squeeze()
        maps_list.append(normalized.byte().tolist())
    return {
        "kind": "feature_maps",
        "featureMaps": {
            "maps": maps_list,
            "channels": C,
            "showing": n_maps,
            "height": target_h,
            "width": target_w,
        },
    }


def _grad_vector(grad: torch.Tensor) -> dict | None:
    """Render gradient as a 1D vector."""
    ft = grad.detach().float().flatten()
    n = ft.numel()
    if n == 0:
        return None
    truncated = n > MAX_VECTOR_ELEMENTS
    values = ft[:MAX_VECTOR_ELEMENTS].tolist() if truncated else ft.tolist()
    return {
        "kind": "vector",
        "vector": {
            "values": [_safe_float(v) for v in values],
            "totalLength": n,
            "truncated": truncated,
        },
    }


# --- Per-type backward viz functions ---

def backward_viz_conv2d(module, activation, gradient):
    result: dict = {"extras": []}

    # Primary viz: gradient feature maps
    if gradient is not None and gradient.dim() == 4:
        result["viz"] = _grad_feature_maps(gradient)

    # Extra 1: Gradient kernels (weight.grad visualized like learned kernels)
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
            "kind": "gradient_kernels",
            "kernels": kernels,
            "showing": n_show,
            "totalFilters": out_c,
            "kernelHeight": kH,
            "kernelWidth": kW,
        })

    # Extra 2: Spatial gradient heatmap (mean |grad| across channels)
    if gradient is not None and gradient.dim() == 4:
        spatial = gradient[0].abs().mean(dim=0).float()  # [H, W]
        H, W = spatial.shape
        smin, smax = float(spatial.min()), float(spatial.max())
        rng = smax - smin if smax > smin else 1.0
        norm = ((spatial - smin) / rng * 255).clamp(0, 255).byte().cpu()
        result["extras"].append({
            "kind": "spatial_gradient_heatmap",
            "pixels": norm.tolist(),
            "height": H,
            "width": W,
        })

    # Extra 3: Per-filter gradient bars
    if weight_grad is not None and weight_grad.dim() == 4:
        per_filter = [_safe_float(float(weight_grad[f].norm())) for f in range(weight_grad.shape[0])]
        result["extras"].append({
            "kind": "per_unit_gradient_bars",
            "values": per_filter,
            "label": "Per-Filter Gradient Norm",
        })

    # Insight
    if weight_grad is not None and weight_grad.dim() == 4:
        norms = [float(weight_grad[f].norm()) for f in range(weight_grad.shape[0])]
        max_idx = norms.index(max(norms))
        result["insight"] = f"Filter {max_idx} has the largest gradient ({norms[max_idx]:.4f}) — it's learning the most from this sample"

    if not result["extras"]:
        del result["extras"]
    return result


def backward_viz_linear(module, activation, gradient):
    result: dict = {"extras": []}

    # Primary viz: per-neuron gradient magnitude
    if gradient is not None:
        if gradient.dim() == 2:
            per_neuron = gradient[0].abs()
        else:
            per_neuron = gradient.abs().flatten()
        result["viz"] = _grad_vector(per_neuron)

    # Extra: gradient weight matrix
    weight_grad = None
    for name, param in module.named_parameters():
        if "weight" in name and param.grad is not None:
            weight_grad = param.grad.detach().cpu().float()
            break
    if weight_grad is not None and weight_grad.dim() == 2:
        actual_rows, actual_cols = weight_grad.shape
        vmin = _safe_float(float(weight_grad.min()))
        vmax = _safe_float(float(weight_grad.max()))
        MAX_DIM = 96
        mat = weight_grad
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
        result["extras"].append({
            "kind": "gradient_weight_matrix",
            "data": mat.tolist(),
            "rows": mat.shape[0],
            "cols": mat.shape[1],
            "actualRows": actual_rows,
            "actualCols": actual_cols,
            "min": vmin,
            "max": vmax,
        })

    # Insight
    if gradient is not None and gradient.dim() >= 2:
        per_neuron = gradient[0].abs() if gradient.dim() == 2 else gradient.abs()
        max_idx = int(per_neuron.argmax())
        max_val = float(per_neuron[max_idx])
        result["insight"] = f"Neuron {max_idx} has the largest gradient ({max_val:.4f}) — the loss is most sensitive to its output"

    if not result["extras"]:
        del result["extras"]
    return result


def backward_viz_activation(module, activation, gradient):
    """ReLU, Sigmoid, Tanh, GELU, LeakyReLU, Softmax."""
    result: dict = {}

    # Primary viz: gradient maps/vector (matching forward shape)
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = _grad_feature_maps(gradient)
        else:
            result["viz"] = _grad_vector(gradient)

    # Before/after: activation gradient vs the input to this activation
    if gradient is not None and activation is not None:
        input_grad_approx = gradient  # gradient at output of this layer
        result["extras"] = [{
            "kind": "before_after_histograms",
            "input": _compact_grad_stats(activation),
            "output": _compact_grad_stats(gradient),
        }]

    return result


def backward_viz_relu(module, activation, gradient):
    result = backward_viz_activation(module, activation, gradient)
    if gradient is not None:
        dead = float((gradient.detach() == 0).sum())
        total = gradient.numel()
        result["insight"] = f"{dead/total*100:.0f}% of gradients were killed (zeroed by ReLU's dead zone)"
    return result


def backward_viz_sigmoid(module, activation, gradient):
    result = backward_viz_activation(module, activation, gradient)
    if gradient is not None:
        g = gradient.detach().float()
        result["insight"] = f"Sigmoid compressed gradient range to [{float(g.min()):.4f}, {float(g.max()):.4f}]"
    return result


def backward_viz_tanh(module, activation, gradient):
    result = backward_viz_activation(module, activation, gradient)
    if gradient is not None:
        g = gradient.detach().float()
        result["insight"] = f"Tanh compressed gradient range to [{float(g.min()):.4f}, {float(g.max()):.4f}]"
    return result


def backward_viz_batchnorm(module, activation, gradient):
    result: dict = {}

    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = _grad_feature_maps(gradient)
        else:
            result["viz"] = _grad_vector(gradient)

    # Gamma/beta gradients
    extras = []
    gamma_grad = None
    beta_grad = None
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if "weight" in name:
            gamma_grad = param.grad.detach().cpu().float()
        elif "bias" in name:
            beta_grad = param.grad.detach().cpu().float()
    if gamma_grad is not None or beta_grad is not None:
        extras.append({
            "kind": "per_unit_gradient_bars",
            "values": [_safe_float(float(v)) for v in (gamma_grad if gamma_grad is not None else beta_grad).tolist()],
            "label": "Scale (γ) Gradient per Channel",
        })
    if extras:
        result["extras"] = extras

    if gradient is not None and activation is not None:
        g_std = float(gradient.detach().float().std())
        a_std = float(activation.detach().float().std())
        result["insight"] = f"BatchNorm redistributed gradients — input std {a_std:.4f}, gradient std {g_std:.4f}"

    return result


def backward_viz_pooling(module, activation, gradient):
    result: dict = {}
    if gradient is not None and gradient.dim() == 4:
        spatial = gradient[0].abs().mean(dim=0).float()
        H, W = spatial.shape
        smin, smax = float(spatial.min()), float(spatial.max())
        rng = smax - smin if smax > smin else 1.0
        norm = ((spatial - smin) / rng * 255).clamp(0, 255).byte().cpu()
        result["extras"] = [{
            "kind": "spatial_gradient_heatmap",
            "pixels": norm.tolist(),
            "height": H,
            "width": W,
        }]
        result["viz"] = _grad_feature_maps(gradient)
    return result


def backward_viz_maxpool(module, activation, gradient):
    result = backward_viz_pooling(module, activation, gradient)
    result["insight"] = "Gradients only flow through max-selected positions — other positions receive zero gradient"
    return result


def backward_viz_avgpool(module, activation, gradient):
    result = backward_viz_pooling(module, activation, gradient)
    result["insight"] = "Gradients distributed equally across the pooling window"
    return result


def backward_viz_lstm(module, activation, gradient):
    result: dict = {}
    # Per-timestep gradient magnitude
    if gradient is not None and gradient.dim() == 3:
        # [B, seq, hidden] → per-timestep norm
        per_step = gradient[0].norm(dim=1).detach().float()  # [seq]
        values = [_safe_float(float(v)) for v in per_step.tolist()]
        result["viz"] = {
            "kind": "vector",
            "vector": {"values": values, "totalLength": len(values), "truncated": False},
        }
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
        # [B, seq, embed] or [B, embed] → per-position gradient magnitude
        g = gradient[0] if gradient.dim() == 3 else gradient
        per_token = g.norm(dim=-1).detach().float() if g.dim() == 2 else g.abs().float()
        values = [_safe_float(float(v)) for v in per_token.tolist()]
        result["viz"] = {
            "kind": "vector",
            "vector": {"values": values, "totalLength": len(values), "truncated": False},
        }
        if values:
            max_idx = values.index(max(values))
            result["insight"] = f"Token at position {max_idx} has the largest gradient — it matters most for the prediction"
    return result


def backward_viz_attention(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        result["viz"] = _grad_vector(gradient)
    result["insight"] = "Gradient of attention output — shows which attended positions affected the loss most"
    return result


def backward_viz_loss(module, activation, gradient):
    result: dict = {}
    if activation is not None and isinstance(activation, torch.Tensor):
        val = _safe_float(float(activation.detach().flatten()[0]))
        result["viz"] = {"kind": "scalar", "scalar": {"value": val}}
    result["insight"] = "Backward pass starts here. Gradient of loss w.r.t. itself = 1.0"
    return result


def backward_viz_flatten(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        result["viz"] = _grad_vector(gradient)
    if activation is not None and gradient is not None:
        result["insight"] = f"Gradient reshaped from {list(gradient.shape)} back to spatial layout upstream"
    return result


def backward_viz_dropout(module, activation, gradient):
    result: dict = {}
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = _grad_feature_maps(gradient)
        else:
            result["viz"] = _grad_vector(gradient)
    result["insight"] = "Dropout mask from forward pass determines which gradients are zeroed"
    return result


def backward_viz_structural(module, activation, gradient):
    """Add, Concat, Reshape, Permute, etc."""
    result: dict = {}
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = _grad_feature_maps(gradient)
        else:
            result["viz"] = _grad_vector(gradient)
    return result


def backward_viz_default(module, activation, gradient):
    """Fallback for unknown node types — basic gradient viz."""
    result: dict = {}
    if gradient is not None:
        if gradient.dim() == 4:
            result["viz"] = _grad_feature_maps(gradient)
        elif gradient.dim() >= 1:
            result["viz"] = _grad_vector(gradient)
        result["insight"] = f"Gradient norm: {float(gradient.detach().float().norm()):.6f}"
    return result


# --- Registry ---

BACKWARD_VIZ: dict[str, callable] = {
    "ml.layers.conv2d": backward_viz_conv2d,
    "ml.layers.conv1d": backward_viz_conv2d,
    "ml.layers.conv_transpose2d": backward_viz_conv2d,
    "ml.layers.linear": backward_viz_linear,
    "ml.activations.relu": backward_viz_relu,
    "ml.activations.sigmoid": backward_viz_sigmoid,
    "ml.activations.tanh": backward_viz_tanh,
    "ml.activations.gelu": backward_viz_activation,
    "ml.activations.leaky_relu": backward_viz_activation,
    "ml.activations.softmax": backward_viz_activation,
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
    "ml.layers.flatten": backward_viz_flatten,
    "ml.layers.dropout": backward_viz_dropout,
    "ml.layers.dropout2d": backward_viz_dropout,
    "ml.structural.add": backward_viz_structural,
    "ml.structural.concat": backward_viz_structural,
    "ml.structural.reshape": backward_viz_structural,
    "ml.structural.permute": backward_viz_structural,
    "ml.structural.sequence_pool": backward_viz_structural,
}

# Add all loss nodes
for loss_type in LOSS_NODES:
    BACKWARD_VIZ[loss_type] = backward_viz_loss


def get_backward_viz(node_type: str, module, activation, gradient):
    """Look up the viz function for a node type and call it."""
    fn = BACKWARD_VIZ.get(node_type, backward_viz_default)
    return fn(module, activation, gradient)
