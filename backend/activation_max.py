"""Activation maximization — find input images that maximize a Conv2d filter's activation.

Algorithm (for each filter):
  1. Start with a random noise image of the dataset's input shape
  2. Forward through the graph, capturing the target filter's output via hook
  3. Loss = -mean(target_filter_output)  (negative so backward = ascent)
  4. Backward through graph to input, update input via gradient
  5. Repeat for N iterations
  6. Final input is the "dream" that maximally excites this filter

Works for both top-level and subgraph-inner Conv2d nodes. Returns pixel data
suitable for direct display.

Kept independent of step_through.py — only depends on graph_builder primitives.
"""

from __future__ import annotations
import torch

from graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    has_trained_model,
    get_device,
    OPTIMIZER_NODES,
    LOSS_NODES,
    SUBGRAPH_TYPE,
    SubGraphModule,
    MULTI_INPUT_NODES,
)
from data_loaders import DATA_LOADERS


def activation_maximization(
    graph_data: dict,
    target_node_id: str,
    num_filters: int = 8,
    iterations: int = 25,
    lr: float = 1.0,
) -> dict:
    """Generate dream images that maximize each of the first N filters of target Conv2d."""
    # Build modules (uses trained weights if available via _model_store side effect;
    # fresh weights otherwise)
    with torch.no_grad():
        modules, _, _, nodes, edges = build_and_run_graph(graph_data)

    # Find target module — check top-level and subgraph interiors
    target = _find_target_module(modules, target_node_id)
    if target is None:
        return {"error": f"Target node {target_node_id} not found"}
    if not hasattr(target, 'weight'):
        return {"error": "Target is not a trainable layer"}

    # Find data node to determine input shape
    data_nid = None
    for nid, n in nodes.items():
        if n["type"] in DATA_LOADERS:
            data_nid = nid
            break
    if not data_nid:
        return {"error": "No data node in graph"}

    # Load a sample to get exact shape + labels template
    loader = DATA_LOADERS[nodes[data_nid]["type"]]
    props = {**nodes[data_nid].get("properties", {}), "batchSize": 1}
    sample = loader(props)
    if not isinstance(sample.get("out"), torch.Tensor) or sample["out"].dim() != 4:
        return {"error": "Activation maximization only supports image datasets"}

    dev = get_device()
    input_shape = sample["out"].shape  # [1, C, H, W]
    labels_template = sample.get("labels")
    if isinstance(labels_template, torch.Tensor):
        labels_template = labels_template.to(dev)

    # Set all modules to eval mode (deterministic forward)
    for m in modules.values():
        m.eval()

    # Hook to capture target's output during forward
    captured: dict = {}

    def hook_fn(module, inp, out):
        # For nn.Conv2d this is [B, OC, H, W]
        captured['value'] = out

    handle = target.register_forward_hook(hook_fn)

    dreams = []
    order = topological_sort(nodes, edges)

    try:
        out_channels = int(target.weight.shape[0])
        n_show = min(num_filters, out_channels)
        using_trained = has_trained_model()

        for filter_idx in range(n_show):
            # Initialize with small noise
            dream = (torch.randn(input_shape, device=dev) * 0.1).detach().requires_grad_(True)

            for step in range(iterations):
                captured.clear()
                results: dict[str, dict] = {
                    data_nid: {"out": dream, "labels": labels_template},
                }

                # Forward through graph, stop once we've captured the target
                hit_target = False
                for nid in order:
                    if nid == data_nid:
                        continue
                    n = nodes[nid]
                    ntype = n["type"]
                    if ntype in OPTIMIZER_NODES or ntype in LOSS_NODES:
                        continue
                    mod = modules.get(nid)
                    if mod is None:
                        continue
                    inputs = gather_inputs(nid, edges, results)
                    if ntype == SUBGRAPH_TYPE:
                        sg = mod(**inputs)
                        fk = next(iter(sg), None)
                        if fk:
                            results[nid] = {"out": sg[fk]}
                    elif ntype in MULTI_INPUT_NODES:
                        results[nid] = {"out": mod(**inputs)}
                    elif "in" in inputs:
                        raw = mod(inputs["in"])
                        results[nid] = raw if isinstance(raw, dict) else {"out": raw}
                    if 'value' in captured:
                        hit_target = True
                        break

                if not hit_target or 'value' not in captured:
                    return {"error": "Could not reach target during forward pass"}

                act = captured['value']
                # Maximize mean activation of the target filter (neg loss for ascent)
                if act.dim() == 4:
                    loss = -act[0, filter_idx].mean()
                elif act.dim() == 2:
                    loss = -act[0, filter_idx]
                else:
                    loss = -act.flatten()[filter_idx]

                if dream.grad is not None:
                    dream.grad.zero_()
                loss.backward()

                with torch.no_grad():
                    grad = dream.grad
                    if grad is not None:
                        grad = grad / (grad.norm() + 1e-8)
                        dream.add_(grad, alpha=lr)
                        dream.clamp_(-3, 3)

            # Convert dream to displayable pixels [0, 255]
            img = dream[0].detach().cpu()
            mn, mx = float(img.min()), float(img.max())
            rng = mx - mn if mx > mn else 1.0
            img = ((img - mn) / rng * 255).clamp(0, 255).byte()

            if img.shape[0] == 1:
                dreams.append({
                    "pixels": img[0].tolist(),
                    "channels": 1,
                    "filterIndex": filter_idx,
                })
            else:
                dreams.append({
                    "pixels": img.permute(1, 2, 0).tolist(),
                    "channels": int(img.shape[0]),
                    "filterIndex": filter_idx,
                })

        return {
            "dreams": dreams,
            "totalFilters": out_channels,
            "iterations": iterations,
            "usingTrainedWeights": using_trained,
        }
    finally:
        handle.remove()


def _find_target_module(modules: dict, target_node_id: str):
    """Find module by node_id, checking both top-level and subgraph inner modules."""
    # Top level
    m = modules.get(target_node_id)
    if m is not None:
        return m
    # Subgraph interiors
    for mod in modules.values():
        if isinstance(mod, SubGraphModule):
            safe_key = mod._key_map.get(target_node_id)
            if safe_key and safe_key in mod.inner_modules:
                return mod.inner_modules[safe_key]
    return None
