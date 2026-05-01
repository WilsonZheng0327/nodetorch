"""Linear layer visualization."""

from graph_builder import _safe_float
from .helpers import MAX_VECTOR_ELEMENTS


def forward_viz_linear(node_type, module, input_tensor, output, inputs, out_dict):
    transformation: dict = {"type": "linear"}

    if input_tensor is not None:
        inp = input_tensor.detach().float()
        if inp.dim() >= 2:
            inp = inp[0]
        inp = inp.flatten()
        transformation["inputDim"] = int(inp.numel())
        n = min(MAX_VECTOR_ELEMENTS, inp.numel())
        transformation["inputVector"] = [_safe_float(float(v)) for v in inp[:n].tolist()]
    else:
        transformation["inputDim"] = 0
        transformation["inputVector"] = []

    if output is not None:
        out = output.detach().float()
        if out.dim() >= 2:
            out = out[0]
        out = out.flatten()
        transformation["outputDim"] = int(out.numel())
        n = min(MAX_VECTOR_ELEMENTS, out.numel())
        transformation["outputVector"] = [_safe_float(float(v)) for v in out[:n].tolist()]
    else:
        transformation["outputDim"] = 0
        transformation["outputVector"] = []

    insight = None
    if input_tensor is not None and output is not None:
        insight = f"Linear projection: {input_tensor.shape[-1]} → {output.shape[-1]}"
    return {"transformation": transformation, "insight": insight}
