"""Normalization layer visualizations (BatchNorm, LayerNorm, GroupNorm, InstanceNorm)."""

import torch
from graph_builder import _safe_float
from .helpers import histogram_data


def forward_viz_norm(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "norm"}

    kind_map = {
        "ml.layers.batchnorm2d": "BatchNorm2d",
        "ml.layers.batchnorm1d": "BatchNorm1d",
        "ml.layers.layernorm": "LayerNorm",
        "ml.layers.groupnorm": "GroupNorm",
        "ml.layers.instancenorm2d": "InstanceNorm2d",
    }
    transformation["normKind"] = kind_map.get(node_type, "Norm")

    if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
        transformation["inputHist"] = histogram_data(input_tensor)
    else:
        transformation["inputHist"] = {"bins": [], "counts": [], "mean": 0, "std": 0}

    if output is not None:
        transformation["outputHist"] = histogram_data(output)
    else:
        transformation["outputHist"] = {"bins": [], "counts": [], "mean": 0, "std": 0}

    if module is not None:
        gamma, beta = None, None
        for name, param in module.named_parameters():
            p = param.detach().cpu().float().flatten()
            vals = [_safe_float(float(v)) for v in p[:32].tolist()]
            if "weight" in name:
                gamma = vals
            elif "bias" in name:
                beta = vals
        if gamma:
            transformation["gamma"] = gamma
        if beta:
            transformation["beta"] = beta

    insights = {
        "ml.layers.batchnorm2d": "Normalized per channel to mean~0, std~1, then scaled by learned gamma/beta",
        "ml.layers.batchnorm1d": "Normalized per channel to mean~0, std~1, then scaled by learned gamma/beta",
        "ml.layers.layernorm": "Normalized across features per sample",
        "ml.layers.groupnorm": "Normalized within groups of channels",
        "ml.layers.instancenorm2d": "Normalized per sample, per channel (spatial mean/var)",
    }
    return {"transformation": transformation, "insight": insights.get(node_type, "Normalization layer")}
