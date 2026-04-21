"""Loss landscape visualization.

Compute the loss surface around the current weights, projected onto two random
directions in weight space. Shows 2D slice: loss(θ + α·d1 + β·d2) for α, β ∈ [-R, R].

Uses filter-wise normalization: each layer's direction is scaled to match the
norm of the layer's original weights. This prevents deep layers with small weights
from dominating the direction.

Output is a 2D grid that can be rendered as a heatmap. Center cell (alpha=0, beta=0)
is the actual loss at the current weights.

Based on "Visualizing the Loss Landscape of Neural Nets" (Li et al. 2018).
"""

from __future__ import annotations
import copy
import torch

from graph_builder import (
    build_and_run_graph,
    gather_inputs,
    topological_sort,
    has_trained_model,
    get_trained_modules,
    get_device,
    OPTIMIZER_NODES,
    LOSS_NODES,
    ALL_LOSS_NODES,
)
from data_loaders import DATA_LOADERS
from forward_utils import execute_node


def compute_loss_landscape(
    graph_data: dict,
    grid_size: int = 11,
    alpha_range: float = 1.0,
    batch_size: int = 32,
) -> dict:
    """Compute a 2D grid of loss values around the current weights.

    Returns:
      {
        "grid": [[loss floats ...]],  # grid_size x grid_size
        "alphaRange": float,
        "gridSize": int,
        "centerLoss": float,
        "minLoss": float,
        "maxLoss": float,
        "usedTrainedWeights": bool,
      }
    """
    graph_data = copy.deepcopy(graph_data)

    # GAN, diffusion, and VAE models don't support loss landscape visualization
    for n in graph_data["graph"]["nodes"]:
        if n["type"] in ("ml.gan.noise_input", "ml.loss.gan"):
            return {"error": "Loss landscape is not available for GAN models — the adversarial loss requires running both generator and discriminator"}
        if n["type"] == "ml.diffusion.noise_scheduler":
            return {"error": "Loss landscape is not available for diffusion models — the loss depends on random timestep sampling"}
        if n["type"] == "ml.loss.vae":
            return {"error": "Loss landscape is not available for VAE models — the reparameterization trick introduces stochastic sampling that makes the surface unreliable"}

    # Use trained modules if available
    if has_trained_model():
        modules = get_trained_modules()
        used_trained = True
        # We still need nodes/edges from the current graph
        from graph_builder import _last_run
        if _last_run and "nodes" in _last_run and "edges" in _last_run:
            nodes = _last_run["nodes"]
            edges = _last_run["edges"]
        else:
            # Fresh build to populate nodes/edges
            _, _, _, nodes, edges = build_and_run_graph(graph_data)
    else:
        modules, _, _, nodes, edges = build_and_run_graph(graph_data)
        used_trained = False

    # Find loss node and data node
    loss_nid = None
    data_nid = None
    for nid, n in nodes.items():
        if n["type"] in ALL_LOSS_NODES:
            loss_nid = nid
        if n["type"] in DATA_LOADERS:
            data_nid = nid
    if not loss_nid or not data_nid:
        return {"error": "Graph must have a data node and a loss node"}

    # Load a single evaluation batch
    loader = DATA_LOADERS[nodes[data_nid]["type"]]
    try:
        tensors = loader({**nodes[data_nid].get("properties", {}), "batchSize": batch_size})
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    dev = get_device()
    images = tensors["out"].to(dev)
    labels = tensors.get("labels")
    if isinstance(labels, torch.Tensor):
        labels = labels.to(dev)

    # Freeze BatchNorm running stats (eval mode for deterministic forward)
    for m in modules.values():
        m.eval()

    # Collect trainable parameters (keep references — we'll mutate these)
    params = []
    for m in modules.values():
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    if not params:
        return {"error": "No trainable parameters in graph"}

    # Save originals
    originals = [p.detach().clone() for p in params]

    # Generate two random directions, filter-wise normalized
    # For each param tensor, direction = random * (||original|| / ||random||)
    def _make_direction():
        direction = []
        for orig in originals:
            rand = torch.randn_like(orig)
            if orig.dim() > 1:
                # Filter-wise normalization: normalize per output channel
                for i in range(rand.shape[0]):
                    rn = rand[i].norm() + 1e-10
                    on = orig[i].norm()
                    rand[i] = rand[i] * (on / rn)
            else:
                # 1D tensor (bias): global normalization
                rn = rand.norm() + 1e-10
                on = orig.norm() + 1e-10
                rand = rand * (on / rn)
            direction.append(rand)
        return direction

    d1 = _make_direction()
    d2 = _make_direction()

    # Evaluate loss at one (alpha, beta) point
    def _eval_loss(alpha: float, beta: float) -> float:
        with torch.no_grad():
            for p, orig, v1, v2 in zip(params, originals, d1, d2):
                p.copy_(orig + alpha * v1 + beta * v2)

            # Forward
            results: dict[str, dict] = {data_nid: {"out": images, "labels": labels}}
            order = topological_sort(nodes, edges)
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
                inputs = gather_inputs(nid, edges, results)
                out = execute_node(ntype, mod, inputs)
                if out is not None:
                    results[nid] = out

            loss_tensor = results.get(loss_nid, {}).get("out")
            if loss_tensor is None:
                return float('nan')
            return float(loss_tensor.item())

    try:
        # Sweep the grid
        grid: list[list[float]] = []
        half = (grid_size - 1) / 2
        for i in range(grid_size):
            row = []
            beta = ((i - half) / half) * alpha_range
            for j in range(grid_size):
                alpha = ((j - half) / half) * alpha_range
                row.append(_eval_loss(alpha, beta))
            grid.append(row)

        # Center loss (should be at grid center)
        center = grid[grid_size // 2][grid_size // 2]

        # Find min/max for color scaling
        flat = [v for row in grid for v in row if v == v]  # NaN-safe
        if not flat:
            return {"error": "Loss landscape produced no valid values — the model may need retraining"}
        minv = min(flat) if flat else 0
        maxv = max(flat) if flat else 1

        # Replace NaN/Inf with None for JSON serialization
        import math
        grid = [
            [v if math.isfinite(v) else None for v in row]
            for row in grid
        ]
        center = center if math.isfinite(center) else minv

        return {
            "grid": grid,
            "alphaRange": alpha_range,
            "gridSize": grid_size,
            "centerLoss": center,
            "minLoss": minv,
            "maxLoss": maxv,
            "usedTrainedWeights": used_trained,
        }
    finally:
        # Restore originals
        with torch.no_grad():
            for p, orig in zip(params, originals):
                p.copy_(orig)
