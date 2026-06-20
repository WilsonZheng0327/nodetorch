"""Tensor and parameter statistics for visualization metadata.

Pure helpers (torch only) used by the forward pass, inference, layer detail,
and the training loops to summarize weights, activations, and gradients.
"""

import math
import torch
import torch.nn as nn


def _safe_float(v: float) -> float | None:
    """Convert to JSON-safe float. NaN and Inf become None."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def tensor_info(t: torch.Tensor) -> dict:
    """Extract displayable info from a tensor (not the raw data)."""
    ft = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "mean": _safe_float(float(ft.mean())),
        "std": _safe_float(float(ft.std())) if ft.numel() > 1 else 0.0,
        "min": _safe_float(float(ft.min())),
        "max": _safe_float(float(ft.max())),
    }


def module_weight_info(module: nn.Module) -> dict | None:
    """Extract weight statistics and histogram bins for visualization."""
    params = list(module.parameters())
    if not params:
        return None

    all_weights = torch.cat([p.detach().flatten() for p in params]).float()
    hist = torch.histogram(all_weights.cpu(), bins=30)

    return {
        "mean": _safe_float(float(all_weights.mean())),
        "std": _safe_float(float(all_weights.std())),
        "min": _safe_float(float(all_weights.min())),
        "max": _safe_float(float(all_weights.max())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
    }


def batchnorm_info(module: nn.Module) -> dict | None:
    """Extract running mean/var histograms from BatchNorm layers."""
    rm = getattr(module, 'running_mean', None)
    rv = getattr(module, 'running_var', None)
    if rm is None or rv is None:
        return None

    rm_f = rm.detach().float().cpu()
    rv_f = rv.detach().float().cpu()
    rm_hist = torch.histogram(rm_f, bins=30)
    rv_hist = torch.histogram(rv_f, bins=30)

    return {
        "runningMean": {
            "mean": _safe_float(float(rm_f.mean())),
            "std": _safe_float(float(rm_f.std())) if rm_f.numel() > 1 else 0.0,
            "histBins": [_safe_float(float(x)) for x in rm_hist.bin_edges[:-1]],
            "histCounts": [int(x) for x in rm_hist.hist],
        },
        "runningVar": {
            "mean": _safe_float(float(rv_f.mean())),
            "std": _safe_float(float(rv_f.std())) if rv_f.numel() > 1 else 0.0,
            "histBins": [_safe_float(float(x)) for x in rv_hist.bin_edges[:-1]],
            "histCounts": [int(x) for x in rv_hist.hist],
        },
    }


def gradient_info(module: nn.Module) -> dict | None:
    """Extract gradient statistics and histogram from the last backward pass."""
    grads = [p.grad.detach().flatten() for p in module.parameters() if p.grad is not None]
    if not grads:
        return None

    all_grads = torch.cat(grads).float()
    hist = torch.histogram(all_grads.cpu(), bins=30)

    return {
        "mean": _safe_float(float(all_grads.mean())),
        "std": _safe_float(float(all_grads.std())) if all_grads.numel() > 1 else 0.0,
        "norm": _safe_float(float(all_grads.norm())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
    }


def activation_info(tensor: torch.Tensor) -> dict:
    """Extract activation statistics and histogram bins for visualization."""
    flat = tensor.detach().flatten().float()
    hist = torch.histogram(flat.cpu(), bins=30)

    return {
        "mean": _safe_float(float(flat.mean())),
        "std": _safe_float(float(flat.std())) if flat.numel() > 1 else 0.0,
        "min": _safe_float(float(flat.min())),
        "max": _safe_float(float(flat.max())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
        "sparsity": _safe_float(float((flat == 0).sum()) / flat.numel()),
    }
