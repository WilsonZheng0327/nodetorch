"""Conv2d / Conv1d / ConvTranspose2d visualization."""

import torch
from graph_builder import _safe_float
from .helpers import feature_maps_data, extract_conv_kernels, EMPTY_FMAPS


def forward_viz_conv2d(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "conv2d"}

    transformation["input"] = feature_maps_data(input_tensor[0]) if (
        input_tensor is not None and input_tensor.dim() == 4) else EMPTY_FMAPS
    transformation["output"] = feature_maps_data(output[0]) if (
        output is not None and output.dim() == 4) else EMPTY_FMAPS
    transformation["kernels"] = extract_conv_kernels(module) if module is not None else None

    # Interactive detail
    if (module is not None and input_tensor is not None and input_tensor.dim() == 4
            and output is not None and output.dim() == 4):
        in_channels = int(input_tensor.shape[1])
        n_in = min(in_channels, 16)
        n_out = min(output.shape[1], 16)

        # Raw input values for each input channel
        raw_inputs = []
        for ch in range(n_in):
            raw_ch = input_tensor[0, ch].detach().float()
            raw_inputs.append([[_safe_float(float(v)) for v in row] for row in raw_ch.tolist()])
        transformation["rawInputs"] = raw_inputs
        transformation["rawInputH"] = int(input_tensor.shape[2])
        transformation["rawInputW"] = int(input_tensor.shape[3])

        # Raw output values per filter
        raw_outs = []
        for f in range(n_out):
            ch = output[0, f].detach().float()
            raw_outs.append([[_safe_float(float(v)) for v in row] for row in ch.tolist()])
        transformation["rawOutputs"] = raw_outs

        # Full kernel weights: [filter][input_channel][kH][kW]
        # Conv2d weight: [out_ch, in_ch, kH, kW]
        # ConvTranspose2d weight: [in_ch, out_ch, kH, kW] — transposed!
        weight = None
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() == 4:
                weight = param.detach().cpu().float()
                break
        if weight is not None:
            is_transpose = (node_type == "ml.layers.conv_transpose2d")
            if is_transpose:
                # weight is [in_ch, out_ch, kH, kW] — swap to [out_ch, in_ch, kH, kW]
                weight = weight.permute(1, 0, 2, 3)
            n_filters = min(weight.shape[0], n_out)
            n_in_k = min(weight.shape[1], n_in)
            all_kernels = []
            for f in range(n_filters):
                per_ch_kernels = []
                for ch in range(n_in_k):
                    k = weight[f, ch]
                    per_ch_kernels.append([[_safe_float(float(v)) for v in row] for row in k.tolist()])
                all_kernels.append(per_ch_kernels)
            transformation["allKernels"] = all_kernels  # [filter][in_ch][kH][kW]

        # Bias per filter
        bias = None
        for name, param in module.named_parameters():
            if "bias" in name:
                bias = param.detach().cpu().float()
                break
        if bias is not None:
            transformation["biases"] = [_safe_float(float(v)) for v in bias[:n_out].tolist()]

        # Conv parameters
        if hasattr(module, 'stride'):
            s = module.stride
            transformation["stride"] = list(s) if isinstance(s, tuple) else [s, s]
        if hasattr(module, 'padding'):
            p = module.padding
            transformation["padding"] = list(p) if isinstance(p, tuple) else [p, p]
        transformation["inputChannels"] = in_channels
        transformation["isTranspose"] = (node_type == "ml.layers.conv_transpose2d")

    insight = None
    if input_tensor is not None and output is not None and output.dim() == 4:
        in_c = input_tensor.shape[1] if input_tensor.dim() == 4 else '?'
        out_c = output.shape[1]
        kH = kW = '?'
        if module is not None:
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() == 4:
                    kH, kW = param.shape[2], param.shape[3]
                    break
        if node_type == "ml.layers.conv_transpose2d":
            insight = f"Upsampled via transposed convolution ({out_c} filters, {kH}x{kW}) to {output.shape[-2]}x{output.shape[-1]}"
        else:
            insight = f"Applied {out_c} filters ({kH}x{kW}) across {in_c} input channels"
    return {"transformation": transformation, "insight": insight}
