"""GAN training loop — alternating discriminator and generator updates.

The GAN graph declares architecture via edges:
  NoiseInput → GeneratorBlock → GAN Loss (fake_scores port)
  DataNode  → DiscriminatorBlock → GAN Loss (real_scores port)
  GAN Loss → Optimizer_G, GAN Loss → Optimizer_D

The training loop identifies G and D by graph structure, then manually
runs D on real+fake data and G to fool D, alternating updates.

Key difference from standard training:
  - Two optimizers, two backward passes per batch
  - Discriminator is run twice (on real and fake data) — not a DAG constraint
  - Generator loss is BCE(D(fake), ones) — computed directly, not via the loss node
  - No accuracy metric — replaced by D loss and G loss
  - Periodically generates sample images for visualization
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from graph_builder import (
    get_device,
    _safe_float,
    GAN_NOISE_TYPE,
    SUBGRAPH_TYPE,
    ALL_LOSS_NODES,
    SubGraphModule,
    gather_inputs,
)
from forward_utils import run_forward_pass

from .base import (
    TrainingContext,
    TrainingResult,
    build_optimizer,
    build_scheduler,
    init_weight_norms,
    collect_node_snapshots,
    build_gradient_flow,
    run_final_forward,
)


def _find_gan_components(ctx: TrainingContext) -> dict:
    """Identify generator, discriminator, noise input, and loss nodes by graph structure.

    Returns dict with keys:
      noise_nid: str — the noise input node ID
      gen_nid: str — the generator subgraph block node ID
      disc_nid: str — the discriminator subgraph block node ID
      loss_nid: str — the GAN loss node ID
      gen_order: list[str] — topological order of generator chain (noise → ... → gen block)
    """
    nodes = ctx.nodes
    edges = ctx.edges

    # Find noise input node
    noise_nid = None
    for nid, n in nodes.items():
        if n["type"] == GAN_NOISE_TYPE:
            noise_nid = nid
            break
    if noise_nid is None:
        raise ValueError("No noise input node found in GAN graph")

    # Find GAN loss node
    loss_nid = None
    for nid, n in nodes.items():
        if n["type"] == "ml.loss.gan":
            loss_nid = nid
            break
    if loss_nid is None:
        raise ValueError("No GAN loss node found")

    # Find what feeds into GAN loss fake_scores port — trace backward
    fake_source_nid = None
    real_source_nid = None
    for edge in edges:
        if edge["target"]["nodeId"] == loss_nid:
            if edge["target"]["portId"] == "fake_scores":
                fake_source_nid = edge["source"]["nodeId"]
            elif edge["target"]["portId"] == "real_scores":
                real_source_nid = edge["source"]["nodeId"]

    if fake_source_nid is None:
        raise ValueError("GAN Loss fake_scores port not connected")
    if real_source_nid is None:
        raise ValueError("GAN Loss real_scores port not connected")

    # The generator block is the subgraph block downstream of noise input
    # (or any block in the path from noise to the fake_scores source)
    # The discriminator block is the subgraph block that feeds real_scores
    # and is also used to score fake data

    # For a simple DCGAN layout:
    # NoiseInput → GeneratorBlock → DiscriminatorBlock → GAN Loss
    # DataNode → DiscriminatorBlock → GAN Loss
    # So fake_source_nid IS the discriminator, and real_source_nid is also the discriminator

    # Find all subgraph blocks
    subgraph_blocks = [nid for nid, n in nodes.items() if n["type"] == SUBGRAPH_TYPE]

    # Trace from noise input downstream to find generator block
    gen_nid = None
    disc_nid = None

    # Build adjacency for downstream traversal
    downstream: dict[str, list[str]] = {nid: [] for nid in nodes}
    for edge in edges:
        src = edge["source"]["nodeId"]
        tgt = edge["target"]["nodeId"]
        downstream[src].append(tgt)

    # Find all nodes reachable from noise input
    noise_reachable = set()
    queue = [noise_nid]
    while queue:
        current = queue.pop(0)
        for child in downstream.get(current, []):
            if child not in noise_reachable:
                noise_reachable.add(child)
                queue.append(child)

    # Generator = first subgraph block downstream of noise
    for nid in ctx.order:
        if nid in subgraph_blocks and nid in noise_reachable:
            if gen_nid is None:
                gen_nid = nid
            elif disc_nid is None and nid != gen_nid:
                # Second subgraph block in path = discriminator
                disc_nid = nid

    # If we only found one subgraph in the noise path, the discriminator
    # might not be directly downstream of the generator in the graph edges
    # (because the training loop runs D separately). Look for any other
    # subgraph block.
    if disc_nid is None:
        for nid in subgraph_blocks:
            if nid != gen_nid:
                disc_nid = nid
                break

    if gen_nid is None:
        raise ValueError("Could not identify generator subgraph block")
    if disc_nid is None:
        raise ValueError("Could not identify discriminator subgraph block")

    # Build the generator execution order (nodes between noise and gen block output)
    # Only include nodes that are: (a) the noise input or downstream of it, and
    # (b) NOT the discriminator, loss, or optimizer nodes.
    from graph_builder import OPTIMIZER_NODES
    gen_order = []
    for nid in ctx.order:
        ntype = nodes[nid]["type"]
        if nid == noise_nid:
            gen_order.append(nid)
        elif (nid in noise_reachable
              and nid != disc_nid
              and nid != loss_nid
              and ntype not in ALL_LOSS_NODES
              and ntype not in OPTIMIZER_NODES):
            gen_order.append(nid)

    return {
        "noise_nid": noise_nid,
        "gen_nid": gen_nid,
        "disc_nid": disc_nid,
        "loss_nid": loss_nid,
        "gen_order": gen_order,
    }


def _run_generator(modules, nodes, edges, gen_order, noise_nid, noise_tensor):
    """Run the generator subgraph chain: noise → ... → generated images."""
    results = {noise_nid: {"out": noise_tensor}}

    for nid in gen_order:
        if nid == noise_nid:
            continue
        mod = modules.get(nid)
        if mod is None:
            continue

        inputs = gather_inputs(nid, edges, results)
        ntype = nodes[nid]["type"]

        if ntype == SUBGRAPH_TYPE:
            sg_out = mod(**inputs)
            first_key = next(iter(sg_out), None)
            if first_key:
                results[nid] = {"out": sg_out[first_key]}
        elif "in" in inputs:
            raw = mod(inputs["in"])
            if isinstance(raw, dict):
                results[nid] = raw
            else:
                results[nid] = {"out": raw}

    # Return the output of the last node in gen_order
    last_nid = gen_order[-1]
    return results.get(last_nid, {}).get("out"), results


def _run_discriminator(disc_module, images):
    """Run the discriminator on a batch of images. Returns raw scores (logits)."""
    if isinstance(disc_module, SubGraphModule):
        sg_out = disc_module(**{"in": images})
        first_key = next(iter(sg_out), None)
        if first_key:
            return sg_out[first_key]
        return None
    else:
        return disc_module(images)


def gan_train(ctx: TrainingContext) -> TrainingResult:
    """Run the GAN training loop: alternating D and G updates."""

    dev = get_device()

    # Identify GAN components
    try:
        comp = _find_gan_components(ctx)
    except ValueError as e:
        return TrainingResult(error=str(e))

    noise_nid = comp["noise_nid"]
    gen_nid = comp["gen_nid"]
    disc_nid = comp["disc_nid"]
    loss_nid = comp["loss_nid"]
    gen_order = comp["gen_order"]

    gen_module = ctx.modules[gen_nid]
    disc_module = ctx.modules[disc_nid]
    loss_module = ctx.modules[loss_nid]

    # Get latent dim from noise input node properties
    noise_props = ctx.nodes[noise_nid].get("properties", {})
    latent_dim = noise_props.get("latentDim", 100)

    # Separate parameters
    gen_params = list(gen_module.parameters())
    disc_params = list(disc_module.parameters())

    if not gen_params:
        return TrainingResult(error="Generator has no trainable parameters")
    if not disc_params:
        return TrainingResult(error="Discriminator has no trainable parameters")

    # Create two optimizers — assign based on optimizer node order
    # The first optimizer is for D, the second for G (or detect by name)
    if len(ctx.optimizer_nodes) < 2:
        return TrainingResult(error="GAN requires exactly 2 optimizer nodes")

    # Try to detect which optimizer is for G vs D based on their position
    # Convention: optimizer closer to loss on the "D" side = D optimizer
    opt_d = build_optimizer(ctx.optimizer_nodes[0], disc_params)
    opt_g = build_optimizer(ctx.optimizer_nodes[1], gen_params)

    # Use primary optimizer for epoch count
    epochs = ctx.epochs
    total_batches = len(ctx.train_loader)
    prev_weight_norms = init_weight_norms(ctx.modules)
    epoch_results = []

    # Sample interval for generating preview images
    sample_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            break

        epoch_start = time.time()
        total_d_loss = 0.0
        total_g_loss = 0.0
        n_batches = 0
        last_batch_results: dict = {}

        batch_report_interval = max(1, total_batches // 20)
        for batch_idx, (real_images, _labels) in enumerate(tqdm(
            ctx.train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False, ncols=80,
        )):
            if ctx.cancel_event and ctx.cancel_event.is_set():
                break

            real_images = real_images.to(dev)
            batch_size = real_images.size(0)

            if ctx.on_batch and batch_idx % batch_report_interval == 0:
                ctx.on_batch({"epoch": epoch + 1, "totalEpochs": epochs,
                              "batch": batch_idx + 1, "totalBatches": total_batches})

            # ================================================================
            # Train Discriminator
            # ================================================================
            opt_d.zero_grad()

            # Generate fake images (detach from G so G gradients don't flow)
            noise = torch.randn(batch_size, latent_dim, device=dev)
            with torch.no_grad():
                fake_images, _ = _run_generator(
                    ctx.modules, ctx.nodes, ctx.edges, gen_order, noise_nid, noise)

            if fake_images is None:
                continue

            # Run discriminator on real and fake
            real_scores = _run_discriminator(disc_module, real_images)
            fake_scores = _run_discriminator(disc_module, fake_images.detach())

            if real_scores is None or fake_scores is None:
                continue

            # Compute D loss using the GAN loss module
            d_loss = loss_module(real_scores, fake_scores)
            d_loss.backward()
            opt_d.step()

            # ================================================================
            # Train Generator
            # ================================================================
            opt_g.zero_grad()

            # Generate fresh fake images (with gradient tracking for G)
            noise = torch.randn(batch_size, latent_dim, device=dev)
            fake_images, gen_results = _run_generator(
                ctx.modules, ctx.nodes, ctx.edges, gen_order, noise_nid, noise)

            if fake_images is None:
                continue

            # Run discriminator on fake images (G wants D to think they're real)
            fake_scores_for_g = _run_discriminator(disc_module, fake_images)

            if fake_scores_for_g is None:
                continue

            # G loss: BCE with logits, target = ones (G wants D to output 1 for fakes)
            g_loss = nn.functional.binary_cross_entropy_with_logits(
                fake_scores_for_g, torch.ones_like(fake_scores_for_g))
            g_loss.backward()
            opt_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            n_batches += 1

            # Store last batch results for snapshots
            last_batch_results = {
                noise_nid: {"out": noise},
                gen_nid: {"out": fake_images.detach()},
                disc_nid: {"out": fake_scores_for_g.detach()},
                loss_nid: {"out": d_loss.detach()},
            }
            # Also include data node
            last_batch_results[ctx.data_node_id] = {"out": real_images}

        if n_batches == 0:
            continue

        epoch_time = time.time() - epoch_start
        avg_d_loss = total_d_loss / n_batches
        avg_g_loss = total_g_loss / n_batches

        # Collect node snapshots
        node_snapshots = collect_node_snapshots(
            ctx.modules, last_batch_results, ctx.nodes, ctx.order, prev_weight_norms,
        )
        gradient_flow = build_gradient_flow(node_snapshots, ctx.nodes, ctx.order)

        # Generate sample images for visualization every few epochs
        generated_samples = None
        if (epoch + 1) % sample_interval == 0 or epoch == epochs - 1:
            try:
                with torch.no_grad():
                    sample_noise = torch.randn(16, latent_dim, device=dev)
                    sample_images, _ = _run_generator(
                        ctx.modules, ctx.nodes, ctx.edges, gen_order, noise_nid, sample_noise)
                    if sample_images is not None:
                        # Normalize to 0-1 for display
                        imgs = sample_images.detach().cpu()
                        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
                        # Convert to list of 2D pixel arrays (take first channel)
                        generated_samples = []
                        for i in range(min(16, imgs.size(0))):
                            img = imgs[i]
                            if img.dim() == 3:
                                img = img[0]  # Take first channel
                            pixels = (img * 255).clamp(0, 255).byte().tolist()
                            generated_samples.append(pixels)
            except Exception:
                pass

        # Build epoch result (GAN-specific format)
        epoch_result = {
            "epoch": epoch + 1,
            "totalEpochs": epochs,
            "loss": _safe_float(avg_d_loss),  # D loss as primary loss metric
            "accuracy": 0.0,  # GAN doesn't have accuracy
            "dLoss": _safe_float(avg_d_loss),
            "gLoss": _safe_float(avg_g_loss),
            "valLoss": None,
            "valAccuracy": None,
            "learningRate": _safe_float(opt_d.param_groups[0]["lr"]),
            "time": round(epoch_time, 2),
            "batches": total_batches,
            "samples": n_batches * real_images.size(0) if n_batches > 0 else 0,
            "gradientFlow": gradient_flow,
            "perClassAccuracy": [],
            "nodeSnapshots": node_snapshots,
            "trackedSamples": [],
            "trainingMode": "gan",
        }

        if generated_samples:
            epoch_result["generatedSamples"] = generated_samples

        epoch_results.append(epoch_result)

        if ctx.on_epoch:
            ctx.on_epoch(epoch_result)

    # Final forward pass for display metadata
    final_results, node_results = run_final_forward(ctx, ctx.modules)

    return TrainingResult(
        epoch_results=epoch_results,
        modules=ctx.modules,
        node_results=node_results,
        final_results=final_results,
        confusion_data=None,
        misclassifications=None,
        training_mode="gan",
    )
