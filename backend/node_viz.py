"""Visualization registry for step-through.

Aggregates per-layer viz functions from backend/viz/ modules.
Each layer type has its own file; this module builds the lookup registries.

Adding a new node's viz: create or edit a file in backend/viz/, then register it here.
"""

from __future__ import annotations
import torch.nn as nn

from graph_builder import ALL_LOSS_NODES
from data_loaders import DATA_LOADERS

# Import per-layer viz functions
from viz.conv import forward_viz_conv2d
from viz.linear import forward_viz_linear
from viz.activation import (
    forward_viz_relu, forward_viz_sigmoid, forward_viz_tanh,
    forward_viz_softmax, forward_viz_gelu, forward_viz_leaky_relu,
)
from viz.norm import forward_viz_norm
from viz.pool import forward_viz_pooling, forward_viz_upsample
from viz.flatten import forward_viz_flatten
from viz.dropout import forward_viz_dropout
from viz.loss import forward_viz_loss
from viz.data import forward_viz_data
from viz.pretrained import forward_viz_pretrained_resnet18
from viz.diffusion import forward_viz_noise_scheduler
from viz.misc import (
    forward_viz_embedding, forward_viz_recurrent,
    forward_viz_mha, forward_viz_sdpa,
    forward_viz_structural, forward_viz_default,
)

# ============================================================================
# Forward viz registry
# ============================================================================

FORWARD_VIZ: dict[str, callable] = {
    # Convolutions
    "ml.layers.conv2d": forward_viz_conv2d,
    "ml.layers.conv1d": forward_viz_conv2d,
    "ml.layers.conv_transpose2d": forward_viz_conv2d,
    # Linear
    "ml.layers.linear": forward_viz_linear,
    # Pretrained
    "ml.layers.pretrained_resnet18": forward_viz_pretrained_resnet18,
    # Activations
    "ml.activations.relu": forward_viz_relu,
    "ml.activations.sigmoid": forward_viz_sigmoid,
    "ml.activations.tanh": forward_viz_tanh,
    "ml.activations.softmax": forward_viz_softmax,
    "ml.activations.gelu": forward_viz_gelu,
    "ml.activations.leaky_relu": forward_viz_leaky_relu,
    # Normalization
    "ml.layers.batchnorm2d": forward_viz_norm,
    "ml.layers.batchnorm1d": forward_viz_norm,
    "ml.layers.layernorm": forward_viz_norm,
    "ml.layers.groupnorm": forward_viz_norm,
    "ml.layers.instancenorm2d": forward_viz_norm,
    # Pooling
    "ml.layers.maxpool2d": forward_viz_pooling,
    "ml.layers.maxpool1d": forward_viz_pooling,
    "ml.layers.avgpool2d": forward_viz_pooling,
    "ml.layers.adaptive_avgpool2d": forward_viz_pooling,
    "ml.layers.upsample": forward_viz_upsample,
    # Flatten
    "ml.layers.flatten": forward_viz_flatten,
    # Dropout
    "ml.layers.dropout": forward_viz_dropout,
    "ml.layers.dropout2d": forward_viz_dropout,
    # Sequence / attention
    "ml.layers.embedding": forward_viz_embedding,
    "ml.layers.lstm": forward_viz_recurrent,
    "ml.layers.gru": forward_viz_recurrent,
    "ml.layers.rnn": forward_viz_recurrent,
    "ml.layers.multihead_attention": forward_viz_mha,
    "ml.layers.attention": forward_viz_sdpa,
    # Structural
    "ml.structural.add": forward_viz_structural,
    "ml.structural.concat": forward_viz_structural,
    "ml.structural.reshape": forward_viz_structural,
    "ml.structural.permute": forward_viz_structural,
    "ml.structural.sequence_pool": forward_viz_structural,
    "ml.structural.reparameterize": forward_viz_structural,
    "ml.preprocessing.tokenizer": forward_viz_structural,
}

# Register loss and data nodes dynamically
for _lt in ALL_LOSS_NODES:
    FORWARD_VIZ[_lt] = forward_viz_loss
for _dt in DATA_LOADERS:
    FORWARD_VIZ[_dt] = forward_viz_data
FORWARD_VIZ["ml.gan.noise_input"] = forward_viz_data
FORWARD_VIZ["ml.diffusion.noise_scheduler"] = forward_viz_noise_scheduler
FORWARD_VIZ["ml.diffusion.timestep_embed"] = forward_viz_data


def get_forward_viz(node_type: str, module, input_tensor, output, inputs, out_dict) -> dict:
    """Look up the forward viz function for a node type and call it."""
    fn = FORWARD_VIZ.get(node_type, forward_viz_default)
    return fn(node_type, module, input_tensor, output, inputs, out_dict)


# ============================================================================
# Backward viz — placeholder
# ============================================================================

def get_backward_viz(node_type: str, module, activation, gradient) -> dict:
    """Backward viz placeholder — returns empty dict for now."""
    return {}


# ============================================================================
# Legacy helpers — still used by backprop_sim.py until backward is redesigned
# ============================================================================

import torch
from graph_builder import _safe_float


def compact_stats_with_norm(tensor: torch.Tensor) -> dict:
    ft = tensor.detach().float().flatten()
    if ft.numel() == 0:
        return {}
    return {
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
        "min": _safe_float(float(ft.min())),
        "max": _safe_float(float(ft.max())),
        "norm": _safe_float(float(ft.norm())),
    }


def param_grad_stats(module: nn.Module | None) -> dict | None:
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
