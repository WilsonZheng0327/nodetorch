"""Simulate one forward+backward step and return per-node gradient magnitudes.

Used to animate the flow of gradients backward through the graph — educational
visualization of backpropagation.

Runs a single mini-batch, does forward, computes loss, backward, and reads each
node's accumulated gradient norm. Top-level only — SubGraphModule returns the
aggregate norm across all inner params.
"""

from __future__ import annotations
import torch

from graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    get_device,
    OPTIMIZER_NODES,
    LOSS_NODES,
)
from data_loaders import DATA_LOADERS
from forward_utils import execute_node


def simulate_backprop(graph_data: dict, batch_size: int = 4) -> dict:
    """Run a single forward+backward pass and return per-node gradient magnitudes
    in topological order. Frontend reverses this for the animation."""
    # Build modules (fresh or trained — doesn't matter much for visualization)
    modules, _, _, nodes, edges = build_and_run_graph(graph_data)

    data_nid = None
    loss_nid = None
    for nid, n in nodes.items():
        if n["type"] in DATA_LOADERS:
            data_nid = nid
        if n["type"] in LOSS_NODES:
            loss_nid = nid
    if not data_nid or not loss_nid:
        return {"error": "Graph must have a data node and a loss node"}

    order = topological_sort(nodes, edges)

    # Load a small batch
    loader = DATA_LOADERS[nodes[data_nid]["type"]]
    try:
        props = {**nodes[data_nid].get("properties", {}), "batchSize": batch_size}
        tensors = loader(props)
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    dev = get_device()
    images = tensors["out"].to(dev)
    labels = tensors.get("labels")
    if isinstance(labels, torch.Tensor):
        labels = labels.to(dev)

    # Forward pass (NOT under no_grad — we need gradients)
    for m in modules.values():
        m.train()
        m.zero_grad(set_to_none=False)

    batch_results: dict[str, dict] = {data_nid: {"out": images, "labels": labels}}
    for nid in order:
        if nid == data_nid:
            continue
        n = nodes[nid]
        ntype = n["type"]
        if ntype in OPTIMIZER_NODES:
            continue
        mod = modules.get(nid)
        if mod is None:
            continue
        inputs = gather_inputs(nid, edges, batch_results)
        try:
            out = execute_node(ntype, mod, inputs)
        except Exception as e:
            if ntype in LOSS_NODES:
                return {"error": f"Loss computation failed: {e}"}
            raise
        if out is not None:
            batch_results[nid] = out

    # Backward
    loss_tensor = batch_results.get(loss_nid, {}).get("out")
    if loss_tensor is None:
        return {"error": "Loss did not produce a value — check predictions/labels connections"}

    try:
        loss_tensor.backward()
    except Exception as e:
        return {"error": f"Backward failed: {e}"}

    # Collect per-top-level-node gradient norms in topological order
    flow = []
    for nid in order:
        n = nodes[nid]
        ntype = n["type"]
        if ntype in DATA_LOADERS or ntype in OPTIMIZER_NODES:
            continue
        mod = modules.get(nid)
        if mod is None:
            continue
        grads = [p.grad.detach().flatten() for p in mod.parameters() if p.grad is not None]
        if grads:
            norm = float(torch.cat(grads).float().norm())
            flow.append({"nodeId": nid, "norm": norm})
        else:
            # Still include non-trainable nodes so animation walks through them
            flow.append({"nodeId": nid, "norm": 0.0})

    return {"flow": flow, "loss": float(loss_tensor.detach().item())}
