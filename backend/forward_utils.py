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
import torch
import torch.nn as nn

from graph_builder import (
    LOSS_NODES,
    MULTI_INPUT_NODES,
    SUBGRAPH_TYPE,
    OPTIMIZER_NODES,
    GAN_NOISE_TYPE,
    gather_inputs,
)
from data_loaders import DATA_LOADERS


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


def run_forward_pass(
    modules: dict[str, nn.Module],
    nodes: dict,
    edges: list,
    order: list[str],
    data_inputs: dict[str, dict],
) -> dict[str, dict]:
    """Run a full forward pass through the graph in topological order.

    This is the canonical "execute the whole graph" function. It replaces the
    many inline dispatch loops that were duplicated across train, validate,
    evaluate, and probe code paths.

    Args:
        modules: node_id → nn.Module (from build_and_run_graph or trained)
        nodes: node_id → node dict
        edges: edge list
        order: topological order of node IDs
        data_inputs: pre-filled results for data nodes, e.g.
                     {data_nid: {"out": images, "labels": labels}}

    Returns:
        batch_results: node_id → {port_id: tensor} for all executed nodes
    """
    batch_results: dict[str, dict] = dict(data_inputs)

    for node_id in order:
        if node_id in batch_results:
            continue
        node = nodes[node_id]
        ntype = node["type"]

        if ntype in OPTIMIZER_NODES or ntype in DATA_LOADERS or ntype == GAN_NOISE_TYPE:
            continue

        mod = modules.get(node_id)
        if mod is None:
            continue

        inputs = gather_inputs(node_id, edges, batch_results)
        result = execute_node(ntype, mod, inputs)
        if result is not None:
            batch_results[node_id] = result

    return batch_results
