"""Diffusion model training loop — learn to denoise images.

Training: corrupt clean images with noise at random timestep t, predict the noise.
The MSE loss compares the model's predicted noise with the actual noise added.

Inference (sampling): start from pure noise, iteratively denoise for T steps.

Graph structure:
  [Data] -> [NoiseScheduler] -out-> [Conv layers...] -> [MSE Loss predictions]
                             -noise-> [MSE Loss labels]
  [MSE Loss] -> [Adam]
  [TimestepEmbed] (unconnected — training loop finds it by type)

The training loop:
  1. Gets clean images from DataLoader
  2. Samples random timestep t for each image in the batch
  3. Generates random noise
  4. Creates noisy images via scheduler.add_noise(images, noise, t)
  5. Concatenates a timestep channel (t/T broadcast to spatial dims)
  6. Feeds [noisy_images, timestep_channel] through the model
  7. MSE loss compares model output (predicted noise) with actual noise
  8. Backward + optimizer step

Every few epochs, generates sample images by running the full denoising loop.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from graph_builder import (
    get_device,
    _safe_float,
    OPTIMIZER_NODES,
    ALL_LOSS_NODES,
    DIFFUSION_SCHEDULER_TYPE,
    DIFFUSION_EMBED_TYPE,
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


def _find_diffusion_components(ctx: TrainingContext) -> dict:
    """Identify noise scheduler, timestep embed, loss, and model nodes.

    Returns dict with keys:
      scheduler_nid: str
      scheduler_module: NoiseSchedulerModule
      embed_nid: str | None
      loss_nid: str
      model_order: list[str] — nodes between scheduler output and loss input
    """
    nodes = ctx.nodes
    edges = ctx.edges

    # Find noise scheduler
    scheduler_nid = None
    for nid, n in nodes.items():
        if n["type"] == DIFFUSION_SCHEDULER_TYPE:
            scheduler_nid = nid
            break
    if scheduler_nid is None:
        raise ValueError("No Noise Scheduler node found in diffusion graph")

    scheduler_module = ctx.modules.get(scheduler_nid)
    if scheduler_module is None:
        raise ValueError("Noise Scheduler module not built")

    # Find timestep embed (optional — may not be present)
    embed_nid = None
    for nid, n in nodes.items():
        if n["type"] == DIFFUSION_EMBED_TYPE:
            embed_nid = nid
            break

    # Find loss node
    loss_nid = ctx.loss_node_ids[0] if ctx.loss_node_ids else None
    if loss_nid is None:
        raise ValueError("No loss node found in diffusion graph")

    # Build model execution order: nodes between scheduler and loss,
    # excluding data, scheduler, embed, loss, optimizer
    skip_types = {DIFFUSION_SCHEDULER_TYPE, DIFFUSION_EMBED_TYPE}
    skip_types.update(OPTIMIZER_NODES)
    skip_types.update(ALL_LOSS_NODES)

    model_order = []
    for nid in ctx.order:
        ntype = nodes[nid]["type"]
        if nid == ctx.data_node_id:
            continue
        if nid == scheduler_nid:
            continue
        if ntype in skip_types:
            continue
        # Only include nodes that have a module (trainable/executable)
        if nid in ctx.modules:
            model_order.append(nid)

    return {
        "scheduler_nid": scheduler_nid,
        "scheduler_module": scheduler_module,
        "embed_nid": embed_nid,
        "loss_nid": loss_nid,
        "model_order": model_order,
    }


def _sample(model_modules, model_order, nodes, edges, scheduler, shape, device, num_steps=None):
    """Generate images by iterative denoising (DDPM sampling).

    Args:
        model_modules: dict of node_id -> nn.Module for the denoising model
        model_order: topological order of model nodes
        nodes: all graph nodes
        edges: all graph edges
        scheduler: NoiseSchedulerModule
        shape: output image shape [B, C, H, W] (C = original channels, not C+1)
        device: torch device
        num_steps: number of denoising steps (defaults to scheduler.num_timesteps)
    """
    if num_steps is None:
        num_steps = scheduler.num_timesteps

    x = torch.randn(shape, device=device)

    # Find scheduler node ID
    scheduler_nid = None
    for nid, n in nodes.items():
        if n["type"] == "ml.diffusion.noise_scheduler":
            scheduler_nid = nid
            break

    for t_val in reversed(range(num_steps)):
        t_tensor = torch.full((shape[0],), t_val, device=device, dtype=torch.long)

        # Timestep channel [B, 1, H, W]
        t_normalized = (t_tensor.float() / scheduler.num_timesteps).view(-1, 1, 1, 1)
        t_channel = t_normalized.expand(shape[0], 1, shape[2], shape[3])

        # Run through the model to predict noise
        with torch.no_grad():
            batch_results: dict = {}
            if scheduler_nid:
                batch_results[scheduler_nid] = {"out": x, "noise": torch.zeros(shape, device=device), "timestep": t_channel}

            from forward_utils import execute_node
            for nid in model_order:
                mod = model_modules.get(nid)
                if mod is None:
                    continue
                inputs = gather_inputs(nid, edges, batch_results)
                result = execute_node(nodes[nid]["type"], mod, inputs)
                if result is not None:
                    batch_results[nid] = result

            # Get predicted noise from last model node
            last_nid = model_order[-1] if model_order else None
            predicted_noise = batch_results.get(last_nid, {}).get("out") if last_nid else None
            if predicted_noise is None:
                break

        # DDPM sampling step
        alpha = scheduler.alphas[t_val]
        alpha_bar = scheduler.alpha_cumprod[t_val]
        beta = scheduler.betas[t_val]

        if t_val > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_bar).sqrt()) * predicted_noise) + beta.sqrt() * noise

    return x.clamp(-1, 1)


def diffusion_train(ctx: TrainingContext) -> TrainingResult:
    """Run the diffusion training loop: add noise -> predict noise -> MSE loss."""

    dev = get_device()

    # Identify components
    try:
        comp = _find_diffusion_components(ctx)
    except ValueError as e:
        return TrainingResult(error=str(e))

    scheduler_nid = comp["scheduler_nid"]
    scheduler_module = comp["scheduler_module"]
    embed_nid = comp["embed_nid"]
    loss_nid = comp["loss_nid"]
    model_order = comp["model_order"]

    # Collect all trainable parameters (from model layers only)
    all_params = list(p for nid in model_order for p in ctx.modules[nid].parameters())
    if not all_params:
        return TrainingResult(error="No trainable parameters in denoising model")

    # Create optimizer and scheduler
    primary_opt = ctx.optimizer_nodes[0]
    props = primary_opt["properties"]
    optimizer = build_optimizer(primary_opt, all_params)
    lr_scheduler = build_scheduler(optimizer, props.get("scheduler", "none"), ctx.epochs)

    # State
    prev_weight_norms = init_weight_norms(ctx.modules)
    epoch_results = []
    total_batches = len(ctx.train_loader)

    # Sample interval for generating preview images
    sample_interval = max(1, ctx.epochs // 5)

    for epoch in range(ctx.epochs):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            break

        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        last_batch_results: dict = {}

        batch_report_interval = max(1, total_batches // 20)
        for batch_idx, (images, _labels) in enumerate(tqdm(
            ctx.train_loader, desc=f"Epoch {epoch+1}/{ctx.epochs}", leave=False, ncols=80,
        )):
            if ctx.cancel_event and ctx.cancel_event.is_set():
                break

            images = images.to(dev)
            batch_size = images.size(0)

            if ctx.on_batch and batch_idx % batch_report_interval == 0:
                ctx.on_batch({"epoch": epoch + 1, "totalEpochs": ctx.epochs,
                              "batch": batch_idx + 1, "totalBatches": total_batches})

            optimizer.zero_grad()

            # 1. Sample random timesteps
            t = torch.randint(0, scheduler_module.num_timesteps, (batch_size,), device=dev)

            # 2. Generate random noise
            noise = torch.randn_like(images)

            # 3. Add noise to clean images
            noisy_images = scheduler_module.add_noise(images, noise, t)

            # 4. Create timestep channel [B, 1, H, W]
            t_normalized = (t.float() / scheduler_module.num_timesteps).view(-1, 1, 1, 1)
            t_channel = t_normalized.expand(batch_size, 1, images.shape[2], images.shape[3])

            # 5. Pre-fill batch results: scheduler outputs noisy images, noise target, and timestep channel
            #    The graph's Concat node combines noisy + timestep before feeding into the model.
            data_inputs = {
                ctx.data_node_id: {"out": images, "labels": noise},
                scheduler_nid: {"out": noisy_images, "noise": noise, "timestep": t_channel},
            }

            # 6. Run forward pass through the rest of the graph
            batch_results = run_forward_pass(ctx.modules, ctx.nodes, ctx.edges, ctx.order, data_inputs)

            # 7. Get loss and backpropagate
            loss_tensor = batch_results.get(loss_nid, {}).get("out")
            if loss_tensor is not None:
                loss_tensor.backward()
                if ctx.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, ctx.grad_clip_norm)
                optimizer.step()
                total_loss += loss_tensor.item()
                n_batches += 1

            last_batch_results = batch_results

        if n_batches == 0:
            continue

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / n_batches

        # Collect node snapshots
        node_snapshots = collect_node_snapshots(
            ctx.modules, last_batch_results, ctx.nodes, ctx.order, prev_weight_norms,
        )
        gradient_flow = build_gradient_flow(node_snapshots, ctx.nodes, ctx.order)

        # LR scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Generate sample images for visualization
        generated_samples = None
        if (epoch + 1) % sample_interval == 0 or epoch == ctx.epochs - 1:
            try:
                # Get original image shape from data
                sample_shape = (8, images.shape[1], images.shape[2], images.shape[3])
                with torch.no_grad():
                    sample_images = _sample(
                        ctx.modules, model_order, ctx.nodes, ctx.edges,
                        scheduler_module, sample_shape, dev,
                    )
                    if sample_images is not None:
                        imgs = sample_images.detach().cpu()
                        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)
                        generated_samples = []
                        for i in range(min(8, imgs.size(0))):
                            img = imgs[i]
                            if img.dim() == 3:
                                img = img[0]  # Take first channel for grayscale
                            pixels = (img * 255).clamp(0, 255).byte().tolist()
                            generated_samples.append(pixels)
            except Exception:
                pass

        # Build epoch result
        epoch_result = {
            "epoch": epoch + 1,
            "totalEpochs": ctx.epochs,
            "loss": _safe_float(avg_loss),
            "accuracy": 0.0,  # Diffusion doesn't have accuracy
            "noiseLoss": _safe_float(avg_loss),
            "valLoss": None,
            "valAccuracy": None,
            "learningRate": _safe_float(current_lr),
            "time": round(epoch_time, 2),
            "batches": total_batches,
            "samples": n_batches * batch_size if n_batches > 0 else 0,
            "gradientFlow": gradient_flow,
            "perClassAccuracy": [],
            "nodeSnapshots": node_snapshots,
            "trackedSamples": [],
            "trainingMode": "diffusion",
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
        training_mode="diffusion",
    )
