"""Dropout visualization."""

import torch
from graph_builder import _safe_float
from .helpers import feature_maps_data, shared_histograms


def forward_viz_dropout(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "dropout"}

    if input_tensor is not None and isinstance(input_tensor, torch.Tensor) and input_tensor.dim() == 4:
        transformation["inputMaps"] = feature_maps_data(input_tensor[0])
    if output is not None and output.dim() == 4:
        transformation["outputMaps"] = feature_maps_data(output[0])

    if input_tensor is not None and isinstance(input_tensor, torch.Tensor) and output is not None:
        inp_hist, out_hist = shared_histograms(input_tensor, output)
        transformation["inputHist"] = inp_hist
        transformation["outputHist"] = out_hist
        inp_flat = input_tensor.detach().float().flatten()
        out_flat = output.detach().float().flatten()
        transformation["inputNonzero"] = int((inp_flat != 0).sum())
        transformation["outputNonzero"] = int((out_flat != 0).sum())
        transformation["totalElements"] = int(inp_flat.numel())

    return {
        "transformation": transformation,
        "insight": "Randomly zeroes values during training (inactive during inference)",
    }
