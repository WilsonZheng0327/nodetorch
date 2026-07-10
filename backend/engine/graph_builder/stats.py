"""Tensor and parameter statistics for visualization metadata.

Pure helpers (torch only) used by the forward pass, inference, layer detail,
and the training loops to summarize weights, activations, and gradients.
"""

import math
import torch
import torch.nn as nn
from typing import TypedDict


# ---- Statistic payload shapes ----
# These TypedDicts document the exact dict shapes the stat helpers below emit as
# visualization metadata (node snapshots, activations, gradients). They are the
# producer-side contract for that data; the frontend reads the same fields ad-hoc
# from the unstructured `metadata` bag by design (see "Key Design Decisions").

SafeFloat = float | None  # NaN/Inf collapse to None (see _safe_float)


class TensorInfo(TypedDict):
    shape: list[int]
    dtype: str
    mean: SafeFloat
    std: SafeFloat
    min: SafeFloat
    max: SafeFloat


class WeightInfo(TypedDict):
    mean: SafeFloat
    std: SafeFloat
    min: SafeFloat
    max: SafeFloat
    histBins: list[SafeFloat]
    histCounts: list[int]


class HistStats(TypedDict):
    """mean/std plus a 30-bin histogram (BatchNorm running mean / running var)."""
    mean: SafeFloat
    std: SafeFloat
    histBins: list[SafeFloat]
    histCounts: list[int]


class BatchNormInfo(TypedDict):
    runningMean: HistStats
    runningVar: HistStats


class GradientInfo(TypedDict):
    mean: SafeFloat
    std: SafeFloat
    norm: SafeFloat
    rms: SafeFloat  # per-parameter magnitude = norm / sqrt(param count)
    histBins: list[SafeFloat]
    histCounts: list[int]


class ActivationInfo(TypedDict):
    mean: SafeFloat
    std: SafeFloat
    min: SafeFloat
    max: SafeFloat
    histBins: list[SafeFloat]
    histCounts: list[int]
    sparsity: SafeFloat


class PredictionInfo(TypedDict, total=False):
    """A single-sample classifier prediction summary (softmax over logits).

    `predictedClass`, `confidence`, and `probabilities` are always present;
    `trueLabelProb` appears only when a true label is supplied and `topK` only
    when `top_k > 0` (see prediction_from_logits)."""
    predictedClass: int
    confidence: SafeFloat
    probabilities: list[SafeFloat]
    trueLabelProb: SafeFloat
    topK: list[dict]


def _safe_float(v: float) -> SafeFloat:
    """Convert to JSON-safe float. NaN and Inf become None."""
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def prediction_from_logits(
    logits: torch.Tensor,
    true_label: int | None = None,
    top_k: int = 0,
) -> PredictionInfo | None:
    """Turn classifier logits into a single-sample prediction summary.

    Reads row 0 of a [batch, classes] tensor: softmax → predicted class, its
    confidence, and the full probability vector. Returns None for non-2D output
    (autoencoders, autoregressive 3D logits, etc). `true_label` adds
    `trueLabelProb`; `top_k > 0` adds a sorted `topK` list of {index, value}.
    """
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2:
        return None
    probs = torch.softmax(logits.detach(), dim=1)[0]
    n_classes = probs.numel()
    pred_class = int(probs.argmax())
    out: PredictionInfo = {
        "predictedClass": pred_class,
        "confidence": _safe_float(float(probs[pred_class])),
        "probabilities": [_safe_float(float(p)) for p in probs.tolist()],
    }
    if true_label is not None:
        out["trueLabelProb"] = (
            _safe_float(float(probs[true_label])) if 0 <= true_label < n_classes else None
        )
    if top_k > 0:
        k = min(top_k, n_classes)
        top = torch.topk(probs, k)
        out["topK"] = [
            {"index": int(i), "value": _safe_float(float(v))}
            for v, i in zip(top.values.tolist(), top.indices.tolist())
        ]
    return out


def tensor_info(t: torch.Tensor) -> TensorInfo:
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


def module_weight_info(module: nn.Module) -> WeightInfo | None:
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


def batchnorm_info(module: nn.Module) -> BatchNormInfo | None:
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


def gradient_info(module: nn.Module) -> GradientInfo | None:
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
        # Root-mean-square gradient = norm / sqrt(param count). Unlike the raw L2
        # norm, this is a per-parameter magnitude, so layers of different sizes
        # are comparable (a big Linear vs a small Conv).
        "rms": _safe_float(float(all_grads.pow(2).mean().sqrt())),
        "histBins": [_safe_float(float(x)) for x in hist.bin_edges[:-1]],
        "histCounts": [int(x) for x in hist.hist],
    }


def activation_info(tensor: torch.Tensor) -> ActivationInfo:
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
