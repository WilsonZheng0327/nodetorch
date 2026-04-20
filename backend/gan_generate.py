"""Generate images from a trained GAN generator.

Feeds random noise through the generator to produce images on demand.
"""

from __future__ import annotations
import torch

from graph_builder import (
    get_device,
    has_trained_model,
    get_trained_modules,
    topological_sort,
    gather_inputs,
    GAN_NOISE_TYPE,
    SUBGRAPH_TYPE,
    OPTIMIZER_NODES,
    ALL_LOSS_NODES,
)
from forward_utils import execute_node
from data_loaders import DENORMALIZERS


def generate_gan_images(graph_data: dict, num_samples: int = 8) -> dict:
    """Generate images by running noise through the trained generator.

    Returns: { images: [sample][y][x] or [sample][y][x][rgb], numSamples, imageH, imageW, channels }
    """
    if not has_trained_model():
        return {"error": "No trained model — train a GAN first"}

    trained = get_trained_modules()
    graph = graph_data["graph"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    edges = graph["edges"]
    order = topological_sort(nodes, edges)

    # Find noise input node
    noise_nid = None
    latent_dim = 100
    for nid, n in nodes.items():
        if n["type"] == GAN_NOISE_TYPE:
            noise_nid = nid
            latent_dim = n.get("properties", {}).get("latentDim", 100)
            break

    if not noise_nid:
        return {"error": "No Noise Input node found — this is for GANs"}

    # Find generator: subgraph block downstream of noise input
    gen_nid = None
    for edge in edges:
        if edge["source"]["nodeId"] == noise_nid:
            target = edge["target"]["nodeId"]
            if nodes[target]["type"] == SUBGRAPH_TYPE:
                gen_nid = target
                break

    # If no subgraph, trace downstream to find all generator nodes
    gen_module = trained.get(gen_nid) if gen_nid else None

    dev = get_device()
    noise = torch.randn(num_samples, latent_dim, device=dev)

    images = []
    with torch.no_grad():
        if gen_module is not None:
            # Generator is a subgraph block — call it directly
            gen_module.eval()
            output = gen_module(**{"in": noise})
            first_key = next(iter(output), None)
            if first_key:
                raw = output[first_key]
            else:
                return {"error": "Generator produced no output"}
            gen_module.train()
        else:
            # Generator is a chain of nodes — run them in order
            skip_types = set(OPTIMIZER_NODES) | ALL_LOSS_NODES | {GAN_NOISE_TYPE}
            gen_order = []
            reachable = set()
            queue = [noise_nid]
            while queue:
                nid = queue.pop(0)
                for edge in edges:
                    if edge["source"]["nodeId"] == nid:
                        target = edge["target"]["nodeId"]
                        if target not in reachable and nodes[target]["type"] not in skip_types:
                            reachable.add(target)
                            queue.append(target)

            # Find the last reachable node before loss/discriminator
            for nid in order:
                if nid in reachable and nid in trained:
                    gen_order.append(nid)

            batch_results = {noise_nid: {"out": noise}}
            for nid in gen_order:
                mod = trained[nid]
                mod.eval()
                inputs = gather_inputs(nid, edges, batch_results)
                result = execute_node(nodes[nid]["type"], mod, inputs)
                if result is not None:
                    batch_results[nid] = result

            last_nid = gen_order[-1] if gen_order else None
            raw = batch_results.get(last_nid, {}).get("out") if last_nid else None
            if raw is None:
                return {"error": "Generator produced no output"}

            for nid in gen_order:
                trained[nid].train()

    # Convert to pixels
    # GAN outputs are typically in [-1, 1] (after Tanh)
    raw = raw.detach().cpu()
    for i in range(raw.shape[0]):
        img = raw[i]
        # Map from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        img = (img.clamp(0, 1) * 255).byte()
        C = img.shape[0]
        if C == 1:
            images.append(img[0].tolist())
        else:
            images.append(img.permute(1, 2, 0).tolist())

    # Get image dims
    sample = raw[0]
    C, H, W = sample.shape

    return {
        "images": images,
        "numSamples": num_samples,
        "imageH": H,
        "imageW": W,
        "channels": C,
    }
