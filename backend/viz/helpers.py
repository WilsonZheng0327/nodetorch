"""Shared helpers for step-through visualizations."""

from __future__ import annotations
import random
import torch
import torch.nn as nn
from graph_builder import _safe_float

MAX_FEATURE_MAPS = 16
MAX_FEATURE_MAP_SIZE = 32
MAX_VECTOR_ELEMENTS = 128
MAX_SCATTER_POINTS = 500
TOP_K_PROBS = 5


def feature_maps_data(fmaps: torch.Tensor) -> dict:
    """Convert [C, H, W] to a feature maps dict (0-255 normalized)."""
    if fmaps.dim() < 3:
        return EMPTY_FMAPS
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
        "maps": maps_list, "channels": C, "showing": n_maps,
        "height": target_h, "width": target_w,
    }


def vector_data(vec: torch.Tensor) -> dict:
    """1D vector preview."""
    ft = vec.detach().float().flatten()
    n = ft.numel()
    truncated = n > MAX_VECTOR_ELEMENTS
    values = ft[:MAX_VECTOR_ELEMENTS].tolist() if truncated else ft.tolist()
    return {"values": [_safe_float(v) for v in values], "totalLength": n}


def histogram_data(tensor: torch.Tensor) -> dict:
    """Histogram + mean/std for a tensor."""
    ft = tensor.detach().float().flatten()
    if ft.numel() == 0:
        return {"bins": [], "counts": [], "mean": 0, "std": 0}
    hist = torch.histogram(ft.cpu(), bins=20)
    return {
        "bins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "counts": [int(x) for x in hist.hist],
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
    }


def shared_histograms(a: torch.Tensor, b: torch.Tensor) -> tuple[dict, dict]:
    """Compute two histograms with shared bin edges for direct comparison."""
    a_flat = a.detach().float().flatten()
    b_flat = b.detach().float().flatten()
    combined = torch.cat([a_flat, b_flat])
    edges = torch.histogram(combined.cpu(), bins=20).bin_edges
    a_hist = torch.histogram(a_flat.cpu(), bins=edges)
    b_hist = torch.histogram(b_flat.cpu(), bins=edges)
    bins = [_safe_float(float(x)) for x in edges[:-1]]
    return (
        {"bins": bins, "counts": [int(x) for x in a_hist.hist],
         "mean": _safe_float(float(a_flat.mean())),
         "std": _safe_float(float(a_flat.std())) if a_flat.numel() > 1 else 0.0},
        {"bins": bins, "counts": [int(x) for x in b_hist.hist],
         "mean": _safe_float(float(b_flat.mean())),
         "std": _safe_float(float(b_flat.std())) if b_flat.numel() > 1 else 0.0},
    )


def scatter_points(input_t: torch.Tensor, output_t: torch.Tensor) -> list[dict]:
    """Sample (input, output) pairs for a scatter plot."""
    inp = input_t.detach().float().flatten()
    out = output_t.detach().float().flatten()
    n = inp.numel()
    if n <= MAX_SCATTER_POINTS:
        indices = range(n)
    else:
        indices = sorted(random.sample(range(n), MAX_SCATTER_POINTS))
    return [{"x": _safe_float(float(inp[i])), "y": _safe_float(float(out[i]))} for i in indices]


def extract_conv_kernels(module) -> dict | None:
    """Conv2d kernels as small images (mean across input channels)."""
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
        "data": kernels, "showing": n_show,
        "totalFilters": out_c, "kernelH": kH, "kernelW": kW,
    }


def default_transformation(output: torch.Tensor) -> dict:
    """Shape-based fallback transformation."""
    result: dict = {"type": "default"}
    if output.numel() == 1:
        result["scalar"] = _safe_float(float(output.flatten()[0]))
    elif output.dim() == 4:
        result["featureMaps"] = feature_maps_data(output[0])
    elif output.dim() >= 1:
        result["vector"] = vector_data(output[0] if output.dim() >= 2 else output)
    return result


# Empty feature maps placeholder
EMPTY_FMAPS = {"maps": [], "channels": 0, "showing": 0, "height": 0, "width": 0}
