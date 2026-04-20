"""Latent space visualization for VAEs.

Generates a grid of decoded images by interpolating across 2D slices of the
latent space. Helps students see how the model organizes concepts smoothly.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from graph_builder import (
    get_device, has_trained_model, get_trained_modules,
    topological_sort, gather_inputs, _safe_float,
    OPTIMIZER_NODES, ALL_LOSS_NODES, MULTI_INPUT_NODES, SUBGRAPH_TYPE,
)
from data_loaders import DATA_LOADERS


def generate_latent_grid(graph_data: dict, grid_size: int = 10, latent_range: float = 3.0) -> dict:
    """Generate a grid of decoded images by sweeping 2 latent dimensions.

    Picks the first two dimensions of the latent space, sweeps from
    -latent_range to +latent_range, decodes each point, returns the pixel grid.

    Args:
        graph_data: serialized graph JSON
        grid_size: number of steps per axis (grid_size × grid_size images)
        latent_range: sweep from -range to +range in each dimension

    Returns:
        { grid: [row][col] of pixel arrays, gridSize, latentRange, imageH, imageW, channels }
    """
    if not has_trained_model():
        return {"error": "No trained model — train a VAE first"}

    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Find the reparameterize node — this is where the latent space lives
    reparam_nid = None
    latent_dim = None
    for nid, n in nodes.items():
        if n["type"] == "ml.structural.reparameterize":
            reparam_nid = nid
            break

    if not reparam_nid:
        return {"error": "No Reparameterize node found — this visualization is for VAEs"}

    # Find the decoder path: nodes downstream of reparam, excluding loss/optimizer
    decoder_order = []
    reachable = set()
    queue = [reparam_nid]
    while queue:
        nid = queue.pop(0)
        for edge in edges:
            if edge["source"]["nodeId"] == nid:
                target = edge["target"]["nodeId"]
                if target not in reachable and nodes[target]["type"] not in OPTIMIZER_NODES and nodes[target]["type"] not in ALL_LOSS_NODES:
                    reachable.add(target)
                    queue.append(target)

    # Order decoder nodes topologically
    for nid in order:
        if nid in reachable:
            decoder_order.append(nid)

    if not decoder_order:
        return {"error": "No decoder path found downstream of Reparameterize"}

    # Determine latent dim from the reparameterize node's connected mean linear
    for edge in edges:
        if edge["target"]["nodeId"] == reparam_nid and edge["target"]["portId"] == "mean":
            mean_nid = edge["source"]["nodeId"]
            mean_node = nodes.get(mean_nid, {})
            latent_dim = mean_node.get("properties", {}).get("outFeatures", 32)
            break

    if latent_dim is None:
        latent_dim = 32  # fallback

    dev = get_device()

    # Generate grid of latent vectors
    # Sweep first two dims, keep the rest at 0
    values = torch.linspace(-latent_range, latent_range, grid_size)
    grid_images = []

    for mod in trained.values():
        if isinstance(mod, nn.Module):
            mod.eval()

    with torch.no_grad():
        for i, y_val in enumerate(values):
            row_images = []
            for j, x_val in enumerate(values):
                # Create latent vector: dim 0 = x, dim 1 = y, rest = 0
                z = torch.zeros(1, latent_dim, device=dev)
                z[0, 0] = x_val
                z[0, 1] = y_val

                # Run through decoder
                results: dict[str, dict] = {}
                results[reparam_nid] = {"out": z}

                for nid in decoder_order:
                    n = nodes[nid]
                    ntype = n["type"]
                    mod = trained.get(nid)
                    if mod is None:
                        continue

                    inputs = gather_inputs(nid, edges, results)

                    if ntype in MULTI_INPUT_NODES:
                        try:
                            results[nid] = {"out": mod(**{k: v for k, v in inputs.items()})}
                        except Exception:
                            continue
                    elif ntype == SUBGRAPH_TYPE:
                        try:
                            sg_out = mod(**inputs)
                            first_key = next(iter(sg_out), None)
                            if first_key:
                                results[nid] = {"out": sg_out[first_key]}
                        except Exception:
                            continue
                    elif "in" in inputs:
                        try:
                            raw = mod(inputs["in"])
                            if isinstance(raw, dict):
                                results[nid] = raw
                            else:
                                results[nid] = {"out": raw}
                        except Exception:
                            continue

                # Get the final decoder output (last node in decoder_order)
                final_out = None
                for nid in reversed(decoder_order):
                    out = results.get(nid, {}).get("out")
                    if out is not None and isinstance(out, torch.Tensor):
                        final_out = out
                        break

                if final_out is None:
                    row_images.append(None)
                    continue

                # Convert to pixel data
                img = final_out[0].detach().cpu()  # [C, H, W]
                img = (img.clamp(0, 1) * 255).byte()
                C = img.shape[0]
                if C == 1:
                    row_images.append(img[0].tolist())
                else:
                    row_images.append(img.permute(1, 2, 0).tolist())

            grid_images.append(row_images)

    # Set back to train mode
    for mod in trained.values():
        if isinstance(mod, nn.Module):
            mod.train()

    # Get image dimensions from the first non-None image
    imageH = imageW = channels = 0
    for row in grid_images:
        for img in row:
            if img is not None:
                if isinstance(img[0][0], list):
                    imageH, imageW, channels = len(img), len(img[0]), len(img[0][0])
                else:
                    imageH, imageW, channels = len(img), len(img[0]), 1
                break
        if imageH > 0:
            break

    return {
        "grid": grid_images,
        "gridSize": grid_size,
        "latentRange": latent_range,
        "latentDim": latent_dim,
        "imageH": imageH,
        "imageW": imageW,
        "channels": channels,
    }
