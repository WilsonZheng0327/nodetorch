"""Shared forward-pass primitives.

execute_node() centralizes how different node types are called — Loss nodes take
named args, MultiInput nodes take **kwargs, subgraph nodes route outputs, layer
nodes take a single "in" tensor, and LSTM/GRU return dicts.

This removes the ~30 lines of dispatch code that was duplicated across:
  - step_through.py (3 places)
  - activation_max.py
  - backprop_sim.py

Kept independent of graph_builder so it can be imported without circular deps.
"""

from __future__ import annotations
import torch.nn as nn

from graph_builder import (
    LOSS_NODES,
    MULTI_INPUT_NODES,
    SUBGRAPH_TYPE,
)


def execute_node(node_type: str, module: nn.Module, inputs: dict) -> dict | None:
    """Execute a single node's module with its gathered inputs.

    Returns a dict of output port → tensor (e.g. {"out": tensor},
    or {"out": ..., "hidden": ..., "cell": ...} for LSTM), or None if
    the node can't be executed (missing required inputs).

    Args:
        node_type: the node's type string (e.g. "ml.layers.conv2d")
        module: pre-built nn.Module for this node
        inputs: dict of {port_id: tensor} from gather_inputs()
    """
    # Loss nodes take (predictions, labels) as positional args
    if node_type in LOSS_NODES:
        if "predictions" in inputs and "labels" in inputs:
            return {"out": module(inputs["predictions"], inputs["labels"])}
        return None

    # Subgraph nodes — returns a dict, we take the first output as the primary
    if node_type == SUBGRAPH_TYPE:
        sg_out = module(**inputs)
        first_key = next(iter(sg_out), None)
        if first_key is None:
            return None
        return {"out": sg_out[first_key]}

    # Multi-input structural nodes (Add, Concat, Attention) — pass all as kwargs
    if node_type in MULTI_INPUT_NODES:
        return {"out": module(**inputs)}

    # Default: single-input layer node
    if "in" in inputs:
        raw = module(inputs["in"])
        if isinstance(raw, dict):
            return raw  # LSTM/GRU return multiple outputs
        return {"out": raw}

    return None
