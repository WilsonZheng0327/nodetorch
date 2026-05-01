"""Pooling and upsampling visualizations."""

from graph_builder import _safe_float
from .helpers import feature_maps_data, EMPTY_FMAPS


def forward_viz_pooling(node_type, module, input_tensor, output, inputs, out_dict):
    kind_map = {
        "ml.layers.maxpool2d": "max", "ml.layers.maxpool1d": "max",
        "ml.layers.avgpool2d": "avg", "ml.layers.adaptive_avgpool2d": "adaptive_avg",
    }
    pool_kind = kind_map.get(node_type, "max")
    transformation: dict = {"type": "pool", "poolKind": pool_kind}

    transformation["input"] = feature_maps_data(input_tensor[0]) if (
        input_tensor is not None and input_tensor.dim() == 4) else EMPTY_FMAPS
    transformation["output"] = feature_maps_data(output[0]) if (
        output is not None and output.dim() == 4) else EMPTY_FMAPS

    # Interactive detail: raw values for one channel + pool params
    if input_tensor is not None and input_tensor.dim() == 4 and output is not None and output.dim() == 4:
        # Raw values for first channel
        raw_in = input_tensor[0, 0].detach().float()
        transformation["rawInput"] = [[_safe_float(float(v)) for v in row] for row in raw_in.tolist()]
        transformation["rawInputH"] = int(raw_in.shape[0])
        transformation["rawInputW"] = int(raw_in.shape[1])

        raw_out = output[0, 0].detach().float()
        transformation["rawOutput"] = [[_safe_float(float(v)) for v in row] for row in raw_out.tolist()]

        # Per-channel scalar values when output is spatially collapsed (1x1 or 2x2)
        if output.shape[2] <= 2 and output.shape[3] <= 2:
            n_ch = min(int(output.shape[1]), 16)
            transformation["channelValues"] = [
                _safe_float(float(output[0, ch].mean())) for ch in range(n_ch)
            ]

        # Pool parameters
        if hasattr(module, 'kernel_size'):
            ks = module.kernel_size
            transformation["poolSize"] = list(ks) if isinstance(ks, tuple) else [ks, ks]
        if hasattr(module, 'stride'):
            s = module.stride
            transformation["strideSize"] = list(s) if isinstance(s, tuple) else [s, s]
        if hasattr(module, 'padding'):
            p = module.padding
            transformation["paddingSize"] = list(p) if isinstance(p, tuple) else [p, p]

    insight = None
    if input_tensor is not None and output is not None and output.dim() == 4:
        before = f"{input_tensor.shape[-2]}x{input_tensor.shape[-1]}"
        after = f"{output.shape[-2]}x{output.shape[-1]}"
        if "adaptive" in node_type:
            insight = f"Adaptive pool to fixed {after}"
        else:
            desc = "Max pooling" if "max" in node_type else "Average pooling"
            insight = f"{desc}: {before} \u2192 {after}"
    return {"transformation": transformation, "insight": insight}


def forward_viz_upsample(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "upsample"}

    transformation["input"] = feature_maps_data(input_tensor[0]) if (
        input_tensor is not None and input_tensor.dim() == 4) else EMPTY_FMAPS
    transformation["output"] = feature_maps_data(output[0]) if (
        output is not None and output.dim() == 4) else EMPTY_FMAPS

    insight = None
    if input_tensor is not None and output is not None and output.dim() == 4:
        insight = f"Upsampled from {input_tensor.shape[-2]}x{input_tensor.shape[-1]} to {output.shape[-2]}x{output.shape[-1]}"
    return {"transformation": transformation, "insight": insight}
