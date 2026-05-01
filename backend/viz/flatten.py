"""Flatten layer visualization."""

from graph_builder import _safe_float
from .helpers import feature_maps_data


def forward_viz_flatten(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "flatten"}

    if input_tensor is not None:
        transformation["inputShape"] = list(input_tensor.shape[1:])
        if input_tensor.dim() == 4:
            transformation["inputMaps"] = feature_maps_data(input_tensor[0])
    else:
        transformation["inputShape"] = []

    if output is not None:
        out_flat = output[0].detach().float().flatten() if output.dim() >= 2 else output.detach().float().flatten()
        transformation["outputLength"] = int(out_flat.numel())
        fmin, fmax = float(out_flat.min()), float(out_flat.max())
        rng = fmax - fmin if fmax != fmin else 1.0
        pixels = ((out_flat - fmin) / rng * 255).clamp(0, 255).byte()
        transformation["flatPixels"] = pixels.tolist()
    else:
        transformation["outputLength"] = 0
        transformation["flatPixels"] = []

    insight = None
    if input_tensor is not None and output is not None:
        dims = list(input_tensor.shape[1:])
        product = 1
        for d in dims:
            product *= d
        dim_str = " x ".join(str(d) for d in dims)
        insight = f"Reshaped [{dim_str}] into a flat vector of {product} values \u2014 same data, different shape"
    return {"transformation": transformation, "insight": insight}
