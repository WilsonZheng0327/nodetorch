"""Miscellaneous node visualizations: embedding, recurrent, attention, structural, default."""

import torch
from graph_builder import _safe_float
from .helpers import default_transformation, feature_maps_data, vector_data, histogram_data, EMPTY_FMAPS


def forward_viz_embedding(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    insight = None
    if output is not None and output.dim() >= 2:
        insight = f"Looked up embeddings of dim {output.shape[-1]} for each token"
    return {"transformation": transformation, "insight": insight}


def forward_viz_recurrent(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    return {"transformation": transformation, "insight": "Processed sequence through recurrent cells"}


def forward_viz_mha(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    return {"transformation": transformation, "insight": "Attention mechanism: weighted sum over positions"}


def forward_viz_sdpa(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    return {"transformation": transformation, "insight": "Attention mechanism: weighted sum over positions"}


def forward_viz_add(node_type, module, input_tensor, output, inputs, out_dict):
    """Add: show all inputs and the element-wise sum."""
    transformation: dict = {"type": "add", "inputs": []}

    # Collect all inputs
    for port_id in sorted(inputs.keys()):
        t = inputs[port_id]
        if not isinstance(t, torch.Tensor):
            continue
        entry: dict = {"label": port_id.upper()}
        if t.dim() == 4:
            entry["featureMaps"] = feature_maps_data(t[0])
        elif t.dim() >= 1:
            entry["vector"] = vector_data(t[0] if t.dim() >= 2 else t)
        transformation["inputs"].append(entry)

    # Output
    if output is not None and isinstance(output, torch.Tensor):
        if output.dim() == 4:
            transformation["output"] = feature_maps_data(output[0])
        else:
            transformation["output"] = None
            transformation["outputVector"] = vector_data(output[0] if output.dim() >= 2 else output)
    else:
        transformation["output"] = None

    n_inputs = len(transformation["inputs"])
    return {"transformation": transformation, "insight": f"Element-wise sum of {n_inputs} inputs (residual / skip connection)"}


def forward_viz_reparameterize(node_type, module, input_tensor, output, inputs, out_dict):
    """VAE reparameterize: show mean + logvar inputs → sampled z output."""
    transformation: dict = {"type": "reparameterize"}

    mean_t = inputs.get("mean")
    logvar_t = inputs.get("logvar")

    if mean_t is not None and isinstance(mean_t, torch.Tensor):
        m = mean_t[0].detach().float().flatten() if mean_t.dim() >= 2 else mean_t.detach().float().flatten()
        transformation["meanValues"] = [_safe_float(float(v)) for v in m[:128].tolist()]
        transformation["meanHist"] = histogram_data(mean_t)
    if logvar_t is not None and isinstance(logvar_t, torch.Tensor):
        lv = logvar_t[0].detach().float().flatten() if logvar_t.dim() >= 2 else logvar_t.detach().float().flatten()
        transformation["logvarValues"] = [_safe_float(float(v)) for v in lv[:128].tolist()]
        transformation["logvarHist"] = histogram_data(logvar_t)
    if output is not None and isinstance(output, torch.Tensor):
        z = output[0].detach().float().flatten() if output.dim() >= 2 else output.detach().float().flatten()
        transformation["zValues"] = [_safe_float(float(v)) for v in z[:128].tolist()]
        transformation["latentDim"] = int(z.numel())

    return {
        "transformation": transformation,
        "insight": "z = mean + exp(0.5 \u00d7 logvar) \u00d7 noise \u2014 samples from the learned latent distribution",
    }


def forward_viz_reshape(node_type, module, input_tensor, output, inputs, out_dict):
    """Reshape: show before and after with shape labels."""
    transformation: dict = {"type": "reshape"}

    if input_tensor is not None and isinstance(input_tensor, torch.Tensor):
        transformation["inputShape"] = list(input_tensor.shape)
        if input_tensor.dim() == 4:
            transformation["inputFmaps"] = feature_maps_data(input_tensor[0])
        elif input_tensor.dim() >= 1:
            transformation["inputVector"] = vector_data(input_tensor[0] if input_tensor.dim() >= 2 else input_tensor)
    else:
        transformation["inputShape"] = []

    if output is not None and isinstance(output, torch.Tensor):
        transformation["outputShape"] = list(output.shape)
        if output.dim() == 4:
            transformation["outputFmaps"] = feature_maps_data(output[0])
        elif output.dim() >= 1:
            transformation["outputVector"] = vector_data(output[0] if output.dim() >= 2 else output)
    else:
        transformation["outputShape"] = []

    in_s = transformation.get("inputShape", [])
    out_s = transformation.get("outputShape", [])
    return {
        "transformation": transformation,
        "insight": f"Reshaped {in_s} \u2192 {out_s} \u2014 same data, different layout",
    }


def _is_constant_tensor(t: torch.Tensor) -> tuple[bool, float]:
    """Check if a tensor has the same value everywhere."""
    flat = t.detach().float().flatten()
    if flat.numel() == 0:
        return False, 0.0
    val = float(flat[0])
    is_const = bool(torch.all(flat == val))
    return is_const, val


def _build_concat_entry(port_id: str, t: torch.Tensor) -> dict:
    """Build a concat input/output entry with smart visualization."""
    entry: dict = {"label": port_id, "shape": list(t.shape)}

    # Check if it's a constant (like timestep channel)
    is_const, const_val = _is_constant_tensor(t)
    if is_const:
        entry["isConstant"] = True
        entry["constantValue"] = _safe_float(const_val)
        return entry

    # Normal tensor
    if t.dim() == 4:
        entry["featureMaps"] = feature_maps_data(t[0])
    elif t.dim() == 3:
        entry["vector"] = vector_data(t[0].flatten())
    elif t.dim() >= 1:
        entry["vector"] = vector_data(t[0] if t.dim() >= 2 else t)
    return entry


def forward_viz_concat(node_type, module, input_tensor, output, inputs, out_dict):
    """Concat: show all inputs as columns + the concatenated output."""
    transformation: dict = {"type": "concat", "inputs": []}

    for port_id in sorted(inputs.keys()):
        t = inputs[port_id]
        if not isinstance(t, torch.Tensor):
            continue
        entry = _build_concat_entry(port_id.replace("in_", "Input "), t)
        transformation["inputs"].append(entry)

    # Output
    if output is not None and isinstance(output, torch.Tensor):
        transformation["outputShape"] = list(output.shape)
        if output.dim() == 4:
            transformation["outputFmaps"] = feature_maps_data(output[0])
        elif output.dim() == 3:
            transformation["outputVector"] = vector_data(output[0].flatten())
        else:
            transformation["outputVector"] = vector_data(output[0] if output.dim() >= 2 else output)

    # Get concat dimension
    dim = 1
    if module is not None and hasattr(module, 'dim'):
        dim = module.dim
    transformation["dim"] = dim

    n = len(transformation["inputs"])
    return {"transformation": transformation, "insight": f"Concatenated {n} inputs along dimension {dim}"}


def forward_viz_structural(node_type, module, input_tensor, output, inputs, out_dict):
    if node_type == "ml.structural.add":
        return forward_viz_add(node_type, module, input_tensor, output, inputs, out_dict)
    if node_type == "ml.structural.concat":
        return forward_viz_concat(node_type, module, input_tensor, output, inputs, out_dict)
    if node_type == "ml.structural.reparameterize":
        return forward_viz_reparameterize(node_type, module, input_tensor, output, inputs, out_dict)
    if node_type == "ml.structural.reshape":
        return forward_viz_reshape(node_type, module, input_tensor, output, inputs, out_dict)

    insights = {
        "ml.structural.permute": "Reordered tensor dimensions",
        "ml.structural.sequence_pool": "Pooled sequence to single vector per sample",
    }
    transformation: dict = {"type": "default"}
    if output is not None:
        transformation = default_transformation(output)
    insight = None
    if node_type in insights:
        insight = insights[node_type]
    return {"transformation": transformation, "insight": insight}


def forward_viz_default(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "default"}
    if output is not None and isinstance(output, torch.Tensor):
        transformation = default_transformation(output)
    return {"transformation": transformation}
