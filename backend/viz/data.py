"""Data node visualization."""

import torch
from data_loaders import DENORMALIZERS
from .helpers import feature_maps_data, vector_data, histogram_data


def forward_viz_data(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "data"}
    if output is not None and isinstance(output, torch.Tensor):
        if output.dim() == 4:
            transformation["featureMaps"] = feature_maps_data(output[0])
        elif output.dim() >= 1:
            transformation["vector"] = vector_data(output[0] if output.dim() >= 2 else output)

        denorm = DENORMALIZERS.get(node_type)
        if denorm is not None and output.dim() == 4:
            raw = denorm(output[0]).clamp(0, 1)
            transformation["rawHist"] = histogram_data(raw)
            transformation["normHist"] = histogram_data(output)

    insight = "Input sample from dataset"
    if node_type in DENORMALIZERS:
        insight = "Input sample \u2014 normalized (shifted and scaled so values center around 0)"
    return {"transformation": transformation, "insight": insight}
