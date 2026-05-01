"""Activation function visualizations (ReLU, Sigmoid, Tanh, GELU, LeakyReLU)."""

import torch
from graph_builder import _safe_float
from .helpers import feature_maps_data, scatter_points, shared_histograms


def forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict):
    """Generic activation — before/after histograms and feature maps."""
    fn_map = {
        "ml.activations.relu": "relu",
        "ml.activations.sigmoid": "sigmoid",
        "ml.activations.tanh": "tanh",
        "ml.activations.gelu": "gelu",
        "ml.activations.leaky_relu": "leaky_relu",
    }
    fn_name = fn_map.get(node_type, "relu")

    inp = input_tensor if isinstance(input_tensor, torch.Tensor) else None
    out = output if isinstance(output, torch.Tensor) else None

    transformation: dict = {"type": "activation", "fn": fn_name, "points": []}

    if inp is not None and out is not None:
        transformation["points"] = scatter_points(inp, out)

    if fn_name == "relu" and out is not None:
        zeros = float((out.detach() == 0).sum())
        transformation["deadFraction"] = _safe_float(zeros / out.numel())

    if fn_name in ("sigmoid", "tanh") and out is not None:
        out_flat = out.detach().float().flatten()
        if fn_name == "sigmoid":
            saturated = float(((out_flat < 0.05) | (out_flat > 0.95)).sum())
        else:
            saturated = float(((out_flat < -0.9) | (out_flat > 0.9)).sum())
        transformation["saturatedFraction"] = _safe_float(saturated / out_flat.numel())

    if inp is not None and inp.dim() == 4:
        transformation["inputMaps"] = feature_maps_data(inp[0])
    if out is not None and out.dim() == 4:
        transformation["outputMaps"] = feature_maps_data(out[0])

    if inp is not None and out is not None:
        inp_hist, out_hist = shared_histograms(inp, out)
        transformation["inputHist"] = inp_hist
        transformation["outputHist"] = out_hist
        combined_flat = torch.cat([inp.detach().float().flatten(), out.detach().float().flatten()])
        transformation["sharedXMin"] = _safe_float(float(combined_flat.min()))
        transformation["sharedXMax"] = _safe_float(float(combined_flat.max()))

    return {"transformation": transformation}


def forward_viz_relu(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    if output is not None and isinstance(output, torch.Tensor):
        zeros = float((output.detach() == 0).sum())
        result["insight"] = f"{zeros/output.numel()*100:.0f}% of values zeroed (ReLU clips negatives to 0)"
    return result


def forward_viz_sigmoid(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    if output is not None and isinstance(output, torch.Tensor):
        out_flat = output.detach().float().flatten()
        saturated = float(((out_flat < 0.05) | (out_flat > 0.95)).sum()) / out_flat.numel()
        result["insight"] = f"Squashed values to (0, 1) — {saturated*100:.0f}% in saturated region"
    return result


def forward_viz_tanh(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    if output is not None and isinstance(output, torch.Tensor):
        out_flat = output.detach().float().flatten()
        saturated = float(((out_flat < -0.9) | (out_flat > 0.9)).sum()) / out_flat.numel()
        result["insight"] = f"Squashed values to (-1, 1) — {saturated*100:.0f}% in saturated region"
    return result


def forward_viz_gelu(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    result["insight"] = "Smooth nonlinearity (like ReLU but with a soft curve near zero)"
    return result


def forward_viz_leaky_relu(node_type, module, input_tensor, output, inputs, out_dict):
    result = forward_viz_activation(node_type, module, input_tensor, output, inputs, out_dict)
    slope = 0.01
    if module is not None and hasattr(module, 'negative_slope'):
        slope = float(module.negative_slope)
    result["transformation"]["negativeSlope"] = slope
    result["insight"] = f"Leaky ReLU with negative slope {slope} — no dead neurons"
    return result


def forward_viz_softmax(node_type, module, input_tensor, output, inputs, out_dict):
    """Softmax: logits → probabilities with top-K."""
    from .helpers import TOP_K_PROBS
    import torch as _torch

    transformation: dict = {"type": "softmax"}

    if input_tensor is not None and isinstance(input_tensor, _torch.Tensor):
        inp = input_tensor.detach().float()
        if inp.dim() >= 2:
            inp = inp[0]
        transformation["logits"] = [_safe_float(float(v)) for v in inp.flatten().tolist()]
    else:
        transformation["logits"] = []

    if output is not None and output.dim() >= 2:
        probs = output[0].detach().float().flatten()
        transformation["probabilities"] = [_safe_float(float(v)) for v in probs.tolist()]
        k = min(TOP_K_PROBS, probs.numel())
        top_vals, top_idx = _torch.topk(probs, k)
        transformation["topK"] = [{"index": int(i), "value": _safe_float(float(v))}
                                   for i, v in zip(top_idx.tolist(), top_vals.tolist())]
    else:
        transformation["probabilities"] = []
        transformation["topK"] = []

    return {"transformation": transformation, "insight": "Converted raw scores to probabilities (sum = 1)"}
