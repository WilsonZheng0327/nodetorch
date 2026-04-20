"""Denoising step-through visualization for diffusion models.

Runs the full DDPM sampling loop and captures the intermediate image
at each timestep. Returns a timeline of denoising steps for the frontend
to display as a scrubable animation.
"""

from __future__ import annotations
import torch
import torch.nn as nn

from graph_builder import (
    get_device,
    has_trained_model,
    get_trained_modules,
    topological_sort,
    gather_inputs,
    _safe_float,
    OPTIMIZER_NODES,
    ALL_LOSS_NODES,
    DIFFUSION_SCHEDULER_TYPE,
    DIFFUSION_EMBED_TYPE,
)
from data_loaders import DATA_LOADERS, DENORMALIZERS


def run_denoise_step_through(graph_data: dict, num_samples: int = 4, capture_every: int = 1) -> dict:
    """Run DDPM denoising and capture the image at each timestep.

    Args:
        graph_data: serialized graph JSON
        num_samples: how many images to generate in parallel
        capture_every: capture a frame every N steps (1 = every step)

    Returns:
        { steps: [{ timestep, pixels: [sample][y][x] or [sample][y][x][rgb] }],
          numTimesteps, imageH, imageW, channels }
    """
    if not has_trained_model():
        return {"error": "No trained model — train a diffusion model first"}

    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Find scheduler
    scheduler_nid = None
    scheduler_module = None
    for nid, n in nodes.items():
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            scheduler_nid = nid
            scheduler_module = trained.get(nid)
            break

    if not scheduler_module:
        return {"error": "No trained noise scheduler found — train a diffusion model first"}

    # Find the data node to determine image shape
    data_node = None
    for n in graph["nodes"]:
        if n["type"] in DATA_LOADERS:
            data_node = n
            break

    # Determine image shape from data node properties or defaults
    if data_node:
        dataset_type = data_node["type"]
        # Load one sample to get shape
        loader = DATA_LOADERS[dataset_type]
        sample = loader({**data_node.get("properties", {}), "batchSize": 1})
        img = sample["out"]
        C, H, W = img.shape[1], img.shape[2], img.shape[3]
    else:
        C, H, W = 1, 28, 28  # default MNIST

    # Find model nodes (between scheduler and loss, excluding scheduler/loss/optimizer/embed)
    skip_types = set(OPTIMIZER_NODES) | ALL_LOSS_NODES | {DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE}
    model_order = []
    model_modules = {}

    # Trace downstream from scheduler
    reachable = set()
    queue = [scheduler_nid]
    while queue:
        nid = queue.pop(0)
        for edge in edges:
            if edge["source"]["nodeId"] == nid:
                target = edge["target"]["nodeId"]
                if target not in reachable and nodes[target]["type"] not in skip_types:
                    reachable.add(target)
                    queue.append(target)

    for nid in order:
        if nid in reachable and nid in trained:
            model_order.append(nid)
            model_modules[nid] = trained[nid]

    if not model_order:
        return {"error": "No denoising model layers found"}

    dev = get_device()
    num_timesteps = scheduler_module.num_timesteps
    shape = (num_samples, C, H, W)

    # Set all modules to eval
    for mod in model_modules.values():
        mod.eval()

    # Start from pure noise
    x = torch.randn(shape, device=dev)
    steps = []

    # Capture initial noise
    steps.append({
        "timestep": num_timesteps,
        "pixels": _tensor_to_pixels_batch(x, data_node["type"] if data_node else None),
    })

    with torch.no_grad():
        for t_val in reversed(range(num_timesteps)):
            t_tensor = torch.full((num_samples,), t_val, device=dev, dtype=torch.long)

            # Build model input: [image, timestep_channel]
            t_normalized = (t_tensor.float() / num_timesteps).view(-1, 1, 1, 1)
            t_channel = t_normalized.expand(num_samples, 1, H, W)
            model_input = torch.cat([x, t_channel], dim=1)

            # Run model forward
            batch_results = {}
            if scheduler_nid:
                batch_results[scheduler_nid] = {"out": model_input, "noise": torch.zeros(shape, device=dev)}

            for nid in model_order:
                mod = model_modules.get(nid)
                if mod is None:
                    continue
                inputs = gather_inputs(nid, edges, batch_results)
                if "in" in inputs:
                    raw = mod(inputs["in"])
                    batch_results[nid] = raw if isinstance(raw, dict) else {"out": raw}

            # Get predicted noise
            last_nid = model_order[-1]
            predicted_noise = batch_results.get(last_nid, {}).get("out")
            if predicted_noise is None:
                break

            # DDPM step
            alpha = scheduler_module.alphas[t_val]
            alpha_bar = scheduler_module.alpha_cumprod[t_val]
            beta = scheduler_module.betas[t_val]
            noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)
            x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * predicted_noise) + beta.sqrt() * noise

            # Capture frame
            if t_val % capture_every == 0 or t_val == 0:
                steps.append({
                    "timestep": t_val,
                    "pixels": _tensor_to_pixels_batch(x.clamp(-1, 1), data_node["type"] if data_node else None),
                })

    # Set back to train mode
    for mod in model_modules.values():
        mod.train()

    return {
        "steps": steps,
        "numTimesteps": num_timesteps,
        "numSamples": num_samples,
        "imageH": H,
        "imageW": W,
        "channels": C,
    }


def _tensor_to_pixels_batch(images: torch.Tensor, dataset_type: str | None) -> list:
    """Convert [B, C, H, W] tensor to list of pixel arrays for JSON."""
    result = []
    for i in range(images.shape[0]):
        img = images[i].detach().cpu()
        # Map from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = (img.clamp(0, 1) * 255).byte()
        C = img.shape[0]
        if C == 1:
            result.append(img[0].tolist())
        else:
            result.append(img.permute(1, 2, 0).tolist())
    return result
