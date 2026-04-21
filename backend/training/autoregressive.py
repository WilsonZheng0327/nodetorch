"""Autoregressive training loop — character-level language modeling.

Trains a model to predict the next token given all previous tokens (teacher forcing).
Loss is per-token CrossEntropy over the vocabulary. Metric is perplexity (exp(loss)).
Generates short text samples every few epochs to show learning progress.
"""

from __future__ import annotations
import time
import math
import torch
from tqdm import tqdm

from graph_builder import (
    get_device,
    _safe_float,
    OPTIMIZER_NODES,
    ALL_LOSS_NODES,
)
from data_loaders import get_shakespeare_vocab, LM_DATASET_TYPES
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


def _generate_sample(modules, nodes, edges, order, data_node_id, dev,
                     max_tokens: int = 100, temperature: float = 0.8) -> str:
    """Generate a short text sample from the trained model."""
    _, idx2char = get_shakespeare_vocab()
    char2idx, _ = get_shakespeare_vocab()
    vocab_size = len(char2idx)

    # Start with a newline as seed
    seed_char = '\n'
    current_ids = torch.tensor([[char2idx.get(seed_char, 0)]], dtype=torch.long, device=dev)

    generated = [seed_char]

    with torch.no_grad():
        for mod in modules.values():
            if hasattr(mod, 'eval'):
                mod.eval()

        for _ in range(max_tokens):
            # Feed the full sequence through the model
            data_inputs = {data_node_id: {"out": current_ids, "labels": current_ids}}
            results = run_forward_pass(modules, nodes, edges, order, data_inputs)

            # Find the output that looks like logits (shape [B, seq_len, vocab_size])
            logits = None
            for nid in reversed(order):
                if nodes[nid]["type"] in OPTIMIZER_NODES or nodes[nid]["type"] in ALL_LOSS_NODES:
                    continue
                out = results.get(nid, {}).get("out")
                if out is not None and isinstance(out, torch.Tensor) and out.dim() == 3 and out.shape[-1] == vocab_size:
                    logits = out
                    break

            if logits is None:
                break

            # Take logits from the last position
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()

            generated.append(idx2char.get(next_id, '?'))
            # Append to sequence
            next_tensor = torch.tensor([[next_id]], dtype=torch.long, device=dev)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            # Limit context window to avoid OOM
            if current_ids.shape[1] > 256:
                current_ids = current_ids[:, -256:]

        for mod in modules.values():
            if hasattr(mod, 'train'):
                mod.train()

    return ''.join(generated).strip()


def autoregressive_train(ctx: TrainingContext) -> TrainingResult:
    """Run autoregressive training: teacher forcing with per-token CrossEntropy."""

    # Collect all trainable parameters
    all_params = list(p for m in ctx.modules.values() for p in m.parameters())

    # Create optimizer and scheduler
    primary_opt = ctx.optimizer_nodes[0]
    props = primary_opt["properties"]
    optimizer = build_optimizer(primary_opt, all_params)
    scheduler = build_scheduler(optimizer, props.get("scheduler", "none"), ctx.epochs)

    # Loss function — per-token CrossEntropy
    # We'll find vocab_size from the data
    char2idx, _ = get_shakespeare_vocab()
    vocab_size = len(char2idx)
    loss_fn = torch.nn.CrossEntropyLoss()

    # State
    prev_weight_norms = init_weight_norms(ctx.modules)
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    early_stopped = False
    epoch_results = []
    total_batches = len(ctx.train_loader)
    dev = get_device()

    # How often to generate a sample (every N epochs)
    sample_interval = max(1, ctx.epochs // 5)

    for epoch in range(ctx.epochs):
        if ctx.cancel_event and ctx.cancel_event.is_set():
            break

        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0
        last_batch_results: dict = {}

        batch_report_interval = max(1, total_batches // 20)
        for batch_idx, (inputs, targets) in enumerate(tqdm(
            ctx.train_loader, desc=f"Epoch {epoch+1}/{ctx.epochs}", leave=False, ncols=80,
        )):
            if ctx.cancel_event and ctx.cancel_event.is_set():
                break
            inputs, targets = inputs.to(dev), targets.to(dev)

            if ctx.on_batch and batch_idx % batch_report_interval == 0:
                ctx.on_batch({"epoch": epoch + 1, "totalEpochs": ctx.epochs,
                              "batch": batch_idx + 1, "totalBatches": total_batches})

            optimizer.zero_grad()

            # Forward pass
            data_inputs = {ctx.data_node_id: {"out": inputs, "labels": targets}}
            batch_results = run_forward_pass(ctx.modules, ctx.nodes, ctx.edges, ctx.order, data_inputs)

            # Find the prediction tensor (logits) — should be [B, seq_len, vocab_size]
            logits = None
            loss_node_id = ctx.loss_node_ids[0] if ctx.loss_node_ids else None

            # Check if the loss node already computed the loss
            loss_tensor = batch_results.get(loss_node_id, {}).get("out") if loss_node_id else None

            if loss_tensor is None:
                # Loss node didn't fire — compute manually from the last layer's output
                for nid in reversed(ctx.order):
                    if ctx.nodes[nid]["type"] in OPTIMIZER_NODES or ctx.nodes[nid]["type"] in ALL_LOSS_NODES:
                        continue
                    out = batch_results.get(nid, {}).get("out")
                    if out is not None and isinstance(out, torch.Tensor) and out.dim() == 3 and out.shape[-1] == vocab_size:
                        logits = out
                        break

                if logits is not None:
                    # Reshape for CrossEntropy: [B*seq_len, vocab_size] vs [B*seq_len]
                    loss_tensor = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
                else:
                    continue  # Skip batch if no logits found

            # Backward
            loss_tensor.backward()
            if ctx.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(all_params, ctx.grad_clip_norm)
            optimizer.step()
            total_loss += loss_tensor.item()
            n_batches += 1
            last_batch_results = batch_results

        # Epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow

        # Validation
        val_loss = None
        val_perplexity = None
        if ctx.val_loader is not None:
            val_total_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for inputs, targets in ctx.val_loader:
                    if ctx.cancel_event and ctx.cancel_event.is_set():
                        break
                    inputs, targets = inputs.to(dev), targets.to(dev)
                    data_inputs = {ctx.data_node_id: {"out": inputs, "labels": targets}}
                    vr = run_forward_pass(ctx.modules, ctx.nodes, ctx.edges, ctx.order, data_inputs)

                    vl = vr.get(loss_node_id, {}).get("out") if loss_node_id else None
                    if vl is None:
                        for nid in reversed(ctx.order):
                            if ctx.nodes[nid]["type"] in OPTIMIZER_NODES or ctx.nodes[nid]["type"] in ALL_LOSS_NODES:
                                continue
                            out = vr.get(nid, {}).get("out")
                            if out is not None and isinstance(out, torch.Tensor) and out.dim() == 3 and out.shape[-1] == vocab_size:
                                vl = loss_fn(out.view(-1, vocab_size), targets.view(-1))
                                break
                    if vl is not None:
                        val_total_loss += float(vl.item())
                        val_n += 1

            if val_n > 0:
                val_loss = val_total_loss / val_n
                val_perplexity = math.exp(min(val_loss, 20))

        # Early stopping
        if ctx.early_stop_patience > 0 and val_loss is not None:
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= ctx.early_stop_patience:
                    early_stopped = True

        # Snapshots
        node_snapshots = collect_node_snapshots(
            ctx.modules, last_batch_results, ctx.nodes, ctx.order, prev_weight_norms,
        )
        gradient_flow = build_gradient_flow(node_snapshots, ctx.nodes, ctx.order)

        # Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Generate text sample periodically
        generated_text = None
        if (epoch + 1) % sample_interval == 0 or epoch == ctx.epochs - 1:
            try:
                generated_text = _generate_sample(
                    ctx.modules, ctx.nodes, ctx.edges, ctx.order,
                    ctx.data_node_id, dev, max_tokens=150, temperature=0.8,
                )
            except Exception:
                generated_text = None

        # Build epoch result
        epoch_result = {
            "epoch": epoch + 1,
            "totalEpochs": ctx.epochs,
            "loss": _safe_float(avg_loss),
            "perplexity": _safe_float(perplexity),
            "accuracy": 0.0,  # Not applicable for LM
            "valLoss": _safe_float(val_loss) if val_loss is not None else None,
            "valPerplexity": _safe_float(val_perplexity) if val_perplexity is not None else None,
            "valAccuracy": None,
            "learningRate": _safe_float(current_lr),
            "time": round(epoch_time, 2),
            "batches": total_batches,
            "samples": n_batches * ctx.batch_size,
            "gradientFlow": gradient_flow,
            "perClassAccuracy": [],
            "nodeSnapshots": node_snapshots,
            "trackedSamples": [],
            "generatedText": generated_text,
            "trainingMode": "autoregressive",
        }
        epoch_results.append(epoch_result)

        if ctx.on_epoch:
            ctx.on_epoch(epoch_result)

        if early_stopped:
            print(f"Early stopping at epoch {epoch + 1} (patience={ctx.early_stop_patience})")
            break

    # Final forward pass
    final_results, node_results = run_final_forward(ctx, ctx.modules)

    return TrainingResult(
        epoch_results=epoch_results,
        modules=ctx.modules,
        node_results=node_results,
        final_results=final_results,
        confusion_data=None,
        misclassifications=None,
        training_mode="autoregressive",
    )
